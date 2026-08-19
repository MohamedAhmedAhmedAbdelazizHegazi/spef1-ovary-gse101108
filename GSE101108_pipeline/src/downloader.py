"""Robust download of files from NCBI GEO.

The module does not use GEOparse's internal downloader (which on Windows fails
with missing temporary files): files are downloaded with ``requests`` in
streaming mode and then handed to GEOparse as local files.

"""

from __future__ import annotations

import gzip
import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

USER_AGENT = "GSE101108-pipeline/1.0 (python-requests)"
CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass
class DownloadResult:
    """Outcome of the download of a single file."""

    file_name: str
    url: str
    path: str | None
    size_bytes: int = 0
    status: str = "pending"  # downloaded | cached | failed | skipped
    attempts: int = 0
    md5: str | None = None
    integrity_ok: bool | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RemoteFile:
    """File announced by the HTML index of the GEO directory."""

    name: str
    url: str
    size_label: str = ""
    last_modified: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class DownloadError(RuntimeError):
    """Unrecoverable error during download."""


# --------------------------------------------------------------------------- #
# Listing                                                                      #
# --------------------------------------------------------------------------- #


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def list_supplementary_files(
    suppl_url: str, timeout: int = 120, retries: int = 3
) -> list[RemoteFile]:
    """List the files present in the GEO supplementary directory.

    Args:
        suppl_url: directory URL (must end with ``/``).
        timeout: HTTP request timeout in seconds.
        retries: number of attempts on network error.

    Returns:
        List of :class:`RemoteFile` (empty if the directory does not exist).

    Raises:
        DownloadError: if the directory is unreachable after the retries.

    """
    if not suppl_url.endswith("/"):
        suppl_url += "/"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _session() as session:
                response = session.get(suppl_url, timeout=timeout)
            if response.status_code == 404:
                LOGGER.warning("Directory supplementare inesistente: %s", suppl_url)
                return []
            response.raise_for_status()
            return _parse_index_html(response.text, suppl_url)
        except RequestException as exc:  # rete assente, timeout, DNS...
            last_error = exc
            LOGGER.warning(
                "Tentativo %d/%d fallito nel leggere %s: %s",
                attempt,
                retries,
                suppl_url,
                exc,
            )
            time.sleep(min(2**attempt, 10))

    raise DownloadError(
        f"Impossibile leggere la directory {suppl_url}. "
        f"Verificare la connessione di rete o eventuali proxy aziendali. "
        f"Ultimo errore: {last_error}"
    )


def _parse_index_html(html: str, base_url: str) -> list[RemoteFile]:
    """Extract the files from the HTML index of an NCBI FTP-over-HTTP server."""
    soup = BeautifulSoup(html, "html.parser")
    files: list[RemoteFile] = []
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("?") or href.endswith("/"):
            continue
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        name = href.rsplit("/", 1)[-1]
        if not name or name.lower() == "parent directory":
            continue
        files.append(RemoteFile(name=name, url=base_url + name))

    # size / date are in the raw index text: best effort
    text_lines = soup.get_text("\n").splitlines()
    by_name = {f.name: f for f in files}
    for line in text_lines:
        for name, remote in by_name.items():
            if line.startswith(name):
                rest = line[len(name):].split()
                if len(rest) >= 2:
                    remote.last_modified = " ".join(rest[:2])
                if rest:
                    remote.size_label = rest[-1]
    LOGGER.info("Trovati %d file supplementari in %s", len(files), base_url)
    for remote in files:
        LOGGER.info("  - %s (%s)", remote.name, remote.size_label or "dimensione n/d")
    return files


def fetch_remote_checksums(
    suppl_url: str, timeout: int = 60
) -> dict[str, str]:
    """Retrieve any MD5 checksums published by NCBI for the series.

    NCBI exposes checksums through the ``filelist``/``md5checksums.txt`` service
    only for some series; if they are unavailable an empty dictionary is returned
    and the integrity check is limited to size and openability.

    """
    checksums: dict[str, str] = {}
    for candidate in ("md5checksums.txt", "filelist.txt"):
        url = suppl_url.rstrip("/") + "/" + candidate
        try:
            with _session() as session:
                response = session.get(url, timeout=timeout)
            if response.status_code != 200:
                continue
            for line in response.text.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 32:
                    checksums[parts[-1].rsplit("/", 1)[-1]] = parts[0].lower()
        except RequestException:
            continue
    if checksums:
        LOGGER.info("Checksum remoti disponibili per %d file", len(checksums))
    else:
        LOGGER.info(
            "Nessun checksum remoto disponibile: verifica limitata a "
            "dimensione e apertura del file"
        )
    return checksums


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - solo verifica di integrita'
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_md5: str | None = None) -> tuple[bool, str]:
    """Check that a downloaded file is usable.

    Verifies that the file exists, is not empty, can be opened (for ``.gz`` files
    the first block is decompressed) and, if available, that the MD5 matches.

    """
    if not path.is_file():
        return False, "file assente"
    if path.stat().st_size == 0:
        return False, "file vuoto"
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as handle:
                if not handle.read(1024):
                    return False, "archivio gzip vuoto"
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            return False, f"archivio gzip corrotto ({exc})"
    else:
        try:
            with path.open("rb") as handle:
                handle.read(1024)
        except OSError as exc:
            return False, f"file non leggibile ({exc})"
    if expected_md5:
        actual = _md5(path)
        if actual.lower() != expected_md5.lower():
            return False, f"checksum non corrispondente ({actual} != {expected_md5})"
        return True, "checksum verificato"
    return True, "dimensione e apertura verificate"


def download_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    retries: int = 3,
    expected_md5: str | None = None,
    show_progress: bool = True,
) -> DownloadResult:
    """Download a file in streaming mode, with retries and integrity check.

    Files that are already present and valid are neither re-downloaded nor
    overwritten.

    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = DownloadResult(
        file_name=destination.name, url=url, path=str(destination), md5=expected_md5
    )

    ok, message = verify_file(destination, expected_md5)
    if ok:
        result.status = "cached"
        result.size_bytes = destination.stat().st_size
        result.integrity_ok = True
        result.message = f"gia' presente ({message})"
        LOGGER.info("File gia' presente e valido, download saltato: %s", destination)
        return result
    if destination.exists():
        LOGGER.warning(
            "File esistente non valido (%s): verra' riscaricato -> %s",
            message,
            destination,
        )

    temp_path = destination.with_suffix(destination.suffix + ".part")
    last_error: str = ""

    for attempt in range(1, retries + 1):
        result.attempts = attempt
        try:
            with _session() as session:
                with session.get(url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length", 0)) or None
                    with temp_path.open("wb") as handle:
                        progress = tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=destination.name[:40],
                            disable=not show_progress,
                            leave=False,
                        )
                        with progress:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                progress.update(len(chunk))

            ok, message = verify_file(temp_path, expected_md5)
            if not ok:
                last_error = message
                LOGGER.warning(
                    "Download non valido (%s), tentativo %d/%d", message, attempt, retries
                )
                temp_path.unlink(missing_ok=True)
                time.sleep(min(2**attempt, 10))
                continue

            _replace_file(temp_path, destination)
            result.status = "downloaded"
            result.size_bytes = destination.stat().st_size
            result.integrity_ok = True
            result.message = message
            LOGGER.info(
                "Scaricato %s (%.2f MB) - %s",
                destination.name,
                result.size_bytes / 1e6,
                message,
            )
            return result

        except RequestException as exc:
            last_error = str(exc)
            LOGGER.warning(
                "Errore di rete su %s (tentativo %d/%d): %s", url, attempt, retries, exc
            )
            temp_path.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 10))
        except OSError as exc:  # permessi Windows, disco pieno, ...
            last_error = str(exc)
            LOGGER.error(
                "Errore di scrittura su %s: %s. Verificare i permessi della "
                "cartella e che il file non sia aperto in un altro programma.",
                destination,
                exc,
            )
            break

    result.status = "failed"
    result.integrity_ok = False
    result.message = last_error or "errore sconosciuto"
    LOGGER.error("Download fallito definitivamente: %s (%s)", url, result.message)
    return result


def _replace_file(source: Path, destination: Path) -> None:
    """Replace ``destination`` with ``source``, handling Windows file locks."""
    for attempt in range(3):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            if attempt == 2:
                raise OSError(
                    f"Impossibile scrivere {destination}: {exc}. "
                    f"Chiudere eventuali programmi che stanno usando il file."
                ) from exc
            time.sleep(1.0)


def download_supplementary_files(
    suppl_url: str,
    target_dir: Path,
    timeout: int = 120,
    retries: int = 3,
    show_progress: bool = True,
) -> tuple[list[DownloadResult], list[RemoteFile]]:
    """Download all the supplementary files of the series.

    Returns:
        Tuple ``(results, listed remote files)``.

    """
    remote_files = list_supplementary_files(suppl_url, timeout=timeout, retries=retries)
    if not remote_files:
        LOGGER.warning("Nessun file supplementare disponibile in %s", suppl_url)
        return [], []

    checksums = fetch_remote_checksums(suppl_url, timeout=timeout)
    results: list[DownloadResult] = []
    for remote in remote_files:
        results.append(
            download_file(
                remote.url,
                Path(target_dir) / remote.name,
                timeout=timeout,
                retries=retries,
                expected_md5=checksums.get(remote.name),
                show_progress=show_progress,
            )
        )
    return results, remote_files


def download_geo_metadata_files(
    soft_url: str,
    matrix_url: str,
    target_dir: Path,
    timeout: int = 120,
    retries: int = 3,
    show_progress: bool = True,
) -> dict[str, DownloadResult]:
    """Download the family SOFT file and the series matrix of the series."""
    target_dir = Path(target_dir)
    return {
        "soft": download_file(
            soft_url,
            target_dir / soft_url.rsplit("/", 1)[-1],
            timeout=timeout,
            retries=retries,
            show_progress=show_progress,
        ),
        "series_matrix": download_file(
            matrix_url,
            target_dir / matrix_url.rsplit("/", 1)[-1],
            timeout=timeout,
            retries=retries,
            show_progress=show_progress,
        ),
    }
