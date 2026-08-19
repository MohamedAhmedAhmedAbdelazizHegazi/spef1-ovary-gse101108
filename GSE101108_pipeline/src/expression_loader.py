"""Robust detection and reading of the expression matrix.

The module does not simply pick the first supplementary file: every file is
profiled (size, rows, columns, gene identifiers, share of numeric and missing
values) and given an explicit priority.

"""

from __future__ import annotations

import gzip
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Patterns used to recognize the data type from the file name.
DATA_TYPE_PATTERNS: tuple[tuple[str, str, int], ...] = (
    (r"raw[\W_]*count", "raw_counts", 100),
    (r"\bcounts?\b|count[\W_]*matrix|htseq", "counts", 90),
    (r"\btpm\b", "tpm", 70),
    (r"\bfpkm\b", "fpkm", 60),
    (r"\brpkm\b", "rpkm", 55),
    (r"\bcpm\b", "cpm", 50),
    (r"norm(alized|alised)?", "normalized_expression", 40),
    (r"express", "expression", 35),
    (r"fpkm|rsem|salmon|kallisto", "expression", 30),
)

#: Extensions recognized as tabular.
READABLE_SUFFIXES = {".txt", ".tsv", ".csv", ".tab", ".gz", ".xlsx", ".xls"}

#: Extensions that are definitely not tabular (skipped without trying to read).
NON_TABULAR_SUFFIXES = {
    ".bam", ".bai", ".bw", ".bed", ".pdf", ".png", ".jpg", ".jpeg", ".zip",
    ".tar", ".rar", ".docx", ".pptx", ".cel", ".idat", ".vcf", ".bigwig",
}

ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$", re.IGNORECASE)
PROBE_SEPARATORS = ("\t", ",", ";", "|", " ")


class ExpressionLoadError(RuntimeError):
    """Error in the detection or reading of the expression matrix."""


@dataclass
class FileProfile:
    """Profilo diagnostico di un file supplementare candidato."""

    file_name: str
    path: str
    size_bytes: int = 0
    compressed: bool = False
    separator: str | None = None
    n_rows_sampled: int = 0
    n_columns: int = 0
    n_numeric_columns: int = 0
    gene_column: str | None = None
    gene_id_kind: str = "unknown"
    numeric_fraction: float = 0.0
    missing_percent: float = 0.0
    value_min: float | None = None
    value_max: float | None = None
    looks_integer: bool | None = None
    data_type: str = "unknown"
    priority: int = 0
    score: float = 0.0
    is_candidate: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ExpressionMatrix:
    """Expression matrix, read and cleaned.

    Attributes:
        values: numeric DataFrame, rows = original gene identifiers,
            columns = samples.
        gene_metadata: any non-numeric columns from the original file.
        data_type: ``raw_counts`` | ``tpm`` | ``fpkm`` | ...
        source_file: file the matrix was read from.
        gene_column: name of the column used as gene identifier.
        warnings: diagnostic messages collected during reading.

    """

    values: pd.DataFrame
    gene_metadata: pd.DataFrame
    data_type: str
    source_file: str
    gene_column: str
    separator: str
    warnings: list[str] = field(default_factory=list)

    @property
    def is_count_like(self) -> bool:
        """True if the data are counts (sum aggregation, DESeq2 ok)."""
        return self.data_type in {"raw_counts", "counts"}


# --------------------------------------------------------------------------- #
# Candidate profiling
# --------------------------------------------------------------------------- #


def classify_data_type(file_name: str) -> tuple[str, int]:
    """Infer data type and priority from the file name."""
    name = file_name.lower()
    for pattern, data_type, priority in DATA_TYPE_PATTERNS:
        if re.search(pattern, name):
            return data_type, priority
    return "unknown", 10


def _open_text(path: Path) -> io.TextIOBase:
    """Open a text file, transparently even if gzip-compressed."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _read_sample_lines(path: Path, n_lines: int = 60) -> list[str]:
    """Read the first non-empty, non-comment lines of the file."""
    lines: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.rstrip("\n\r")
            if not stripped.strip():
                continue
            if stripped.lstrip().startswith("#") or stripped.startswith("!"):
                continue
            lines.append(stripped)
            if len(lines) >= n_lines:
                break
    return lines


def detect_separator(lines: Sequence[str]) -> str:
    """Detect the most likely column separator."""
    if not lines:
        return "\t"
    best_sep, best_score = "\t", -1.0
    for sep in PROBE_SEPARATORS:
        counts = [line.count(sep) for line in lines[:20]]
        if not counts or max(counts) == 0:
            continue
        consistent = len(set(counts)) == 1
        score = max(counts) + (10 if consistent else 0)
        if score > best_score:
            best_sep, best_score = sep, score
    return best_sep


def count_comment_lines(path: Path) -> int:
    """Numero di righe iniziali di commento (``#`` o ``!``)."""
    skipped = 0
    with _open_text(path) as handle:
        for line in handle:
            if line.lstrip().startswith(("#", "!")) or not line.strip():
                skipped += 1
            else:
                break
    return skipped


def detect_gene_column(
    columns: Iterable[str], candidates: Sequence[str], frame: pd.DataFrame | None = None
) -> str | None:
    """Locate the column holding the gene identifiers.

    The search proceeds in three steps: exact name (case-insensitive), contained
    name, and finally the content of the non-numeric columns (looking for Ensembl
    identifiers or gene symbols).

    """
    columns = list(columns)
    normalized = {re.sub(r"[\s\-_\.]+", "", str(c)).lower(): c for c in columns}
    for candidate in candidates:
        key = re.sub(r"[\s\-_\.]+", "", candidate).lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = re.sub(r"[\s\-_\.]+", "", candidate).lower()
        for norm, original in normalized.items():
            if key and key in norm:
                return original
    if frame is not None:
        for column in columns:
            series = frame[column].astype(str).head(50)
            if series.empty:
                continue
            ensembl_hits = series.str.match(ENSEMBL_RE).mean()
            symbol_hits = series.str.match(r"^[A-Za-z][A-Za-z0-9\-\.@]{1,20}$").mean()
            if ensembl_hits > 0.6 or symbol_hits > 0.8:
                return column
    # last resort: unnamed index in the first column
    if columns and (str(columns[0]).strip() == "" or str(columns[0]).startswith("Unnamed")):
        return columns[0]
    return None


def profile_file(path: Path, gene_column_candidates: Sequence[str]) -> FileProfile:
    """Analyze a supplementary file and compute its candidacy score."""
    path = Path(path)
    data_type, priority = classify_data_type(path.name)
    profile = FileProfile(
        file_name=path.name,
        path=str(path),
        size_bytes=path.stat().st_size if path.is_file() else 0,
        compressed=path.suffix == ".gz",
        data_type=data_type,
        priority=priority,
    )

    inner_suffix = Path(path.name[:-3]).suffix if path.suffix == ".gz" else path.suffix
    if inner_suffix.lower() in NON_TABULAR_SUFFIXES:
        profile.reason = f"estensione non tabellare ({inner_suffix})"
        return profile
    if profile.size_bytes == 0:
        profile.reason = "file vuoto"
        return profile

    try:
        lines = _read_sample_lines(path)
        if not lines:
            profile.reason = "nessuna riga di dati leggibile"
            return profile
        separator = detect_separator(lines)
        profile.separator = separator
        frame = pd.read_csv(
            io.StringIO("\n".join(lines)),
            sep=separator,
            engine="python",
            comment=None,
        )
    except (OSError, UnicodeDecodeError, gzip.BadGzipFile) as exc:
        profile.reason = f"file illeggibile o corrotto: {exc}"
        return profile
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        profile.reason = f"formato non riconosciuto: {exc}"
        return profile

    profile.n_rows_sampled = len(frame)
    profile.n_columns = frame.shape[1]
    if frame.shape[1] < 2:
        profile.reason = "meno di due colonne: non e' una matrice"
        return profile

    gene_column = detect_gene_column(frame.columns, gene_column_candidates, frame)
    profile.gene_column = gene_column
    if gene_column is None:
        profile.reason = "colonna degli identificativi genici non individuata"
        return profile

    data = frame.drop(columns=[gene_column], errors="ignore")
    numeric = data.apply(pd.to_numeric, errors="coerce")
    numeric_columns = [c for c in numeric.columns if numeric[c].notna().mean() > 0.8]
    profile.n_numeric_columns = len(numeric_columns)
    total_cells = max(numeric.size, 1)
    profile.numeric_fraction = float(numeric.notna().sum().sum() / total_cells)

    if not numeric_columns:
        profile.reason = "nessuna colonna numerica: non e' una matrice di espressione"
        return profile

    numeric_block = numeric[numeric_columns]
    profile.missing_percent = float(
        numeric_block.isna().sum().sum() / max(numeric_block.size, 1) * 100
    )
    finite = numeric_block.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        profile.value_min = float(np.nanmin(finite))
        profile.value_max = float(np.nanmax(finite))
        profile.looks_integer = bool(np.allclose(finite, np.round(finite)))

    ids = frame[gene_column].astype(str)
    if ids.str.match(ENSEMBL_RE).mean() > 0.6:
        profile.gene_id_kind = "ensembl"
    elif ids.str.match(r"^\d+$").mean() > 0.8:
        profile.gene_id_kind = "entrez"
    else:
        profile.gene_id_kind = "symbol_or_other"

    # If the name says nothing, integer data suggest counts.
    if profile.data_type == "unknown" and profile.looks_integer:
        profile.data_type = "counts"
        profile.priority = 80

    profile.is_candidate = True
    profile.score = (
        profile.priority
        + min(profile.n_numeric_columns, 200) * 0.1
        + profile.numeric_fraction * 10
        - profile.missing_percent * 0.1
    )
    profile.reason = (
        f"matrice valida: {profile.n_numeric_columns} colonne numeriche, "
        f"id genici '{profile.gene_id_kind}', tipo '{profile.data_type}'"
    )
    return profile


def profile_directory(
    directory: Path, gene_column_candidates: Sequence[str]
) -> list[FileProfile]:
    """Profile all the files present in the raw-data directory."""
    directory = Path(directory)
    profiles: list[FileProfile] = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix == ".part":
            continue
        if "family.soft" in path.name or "series_matrix" in path.name:
            continue  # metadata, not matrices
        profiles.append(profile_file(path, gene_column_candidates))
    for profile in profiles:
        LOGGER.info(
            "Candidato %-45s tipo=%-12s score=%6.1f  %s",
            profile.file_name,
            profile.data_type,
            profile.score,
            profile.reason,
        )
    return profiles


def select_best_candidate(profiles: Sequence[FileProfile]) -> FileProfile:
    """Choose the matrix to use among the valid candidates.

    Raises:
        ExpressionLoadError: if no file is usable as a matrix.

    """
    candidates = [p for p in profiles if p.is_candidate]
    if not candidates:
        details = "; ".join(f"{p.file_name}: {p.reason}" for p in profiles) or "nessun file"
        raise ExpressionLoadError(
            "Nessun file supplementare utilizzabile come matrice di espressione. "
            f"Dettagli: {details}. Controllare la cartella data/raw e, se "
            "necessario, fornire manualmente la matrice."
        )
    best = max(candidates, key=lambda p: (p.score, p.n_numeric_columns))
    LOGGER.info(
        "Matrice selezionata: %s (tipo=%s, score=%.1f) perche' ha la priorita' "
        "piu' alta fra i %d candidati validi",
        best.file_name,
        best.data_type,
        best.score,
        len(candidates),
    )
    return best


def candidates_table(profiles: Sequence[FileProfile], selected: str | None) -> pd.DataFrame:
    """Summary table of the candidate files for the report."""
    rows = []
    for profile in profiles:
        row = profile.to_dict()
        row["selected"] = profile.file_name == selected
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Lettura completa                                                             #
# --------------------------------------------------------------------------- #


def read_expression_matrix(
    profile: FileProfile, gene_column_candidates: Sequence[str]
) -> ExpressionMatrix:
    """Read in full the matrix described by ``profile``.

    Handles gzip compression, automatic separator, comment lines, duplicate
    columns and non-numeric columns (treated as gene metadata).

    """
    path = Path(profile.path)
    separator = profile.separator or "\t"
    skiprows = count_comment_lines(path)
    warnings: list[str] = []

    try:
        frame = pd.read_csv(
            path,
            sep=separator,
            skiprows=skiprows,
            compression="gzip" if path.suffix == ".gz" else "infer",
            low_memory=False,
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise ExpressionLoadError(
            f"Impossibile leggere {path}: {exc}. Il file potrebbe essere "
            f"corrotto: cancellarlo e rilanciare la pipeline."
        ) from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ExpressionLoadError(
            f"Formato inatteso in {path}: {exc}. Verificare il separatore "
            f"(rilevato '{separator}') e la presenza di header multipli."
        ) from exc

    if frame.empty:
        raise ExpressionLoadError(f"La matrice {path.name} non contiene righe.")

    frame, dup_msg = _deduplicate_columns(frame)
    if dup_msg:
        warnings.append(dup_msg)

    gene_column = detect_gene_column(frame.columns, gene_column_candidates, frame)
    if gene_column is None:
        raise ExpressionLoadError(
            f"Colonna degli identificativi genici non individuata in {path.name}. "
            f"Colonne disponibili: {list(frame.columns)[:15]}. Aggiungere il nome "
            f"corretto a GENE_ID_COLUMN_CANDIDATES in config.py."
        )

    gene_ids = frame[gene_column].astype(str).str.strip()
    data = frame.drop(columns=[gene_column])

    numeric = data.apply(pd.to_numeric, errors="coerce")
    numeric_mask = numeric.notna().mean() > 0.8
    numeric_columns = [c for c in data.columns if numeric_mask.get(c, False)]
    metadata_columns = [c for c in data.columns if c not in numeric_columns]

    if not numeric_columns:
        raise ExpressionLoadError(
            f"{path.name} non contiene colonne numeriche utilizzabili come "
            f"campioni. Verificare il separatore o la presenza di header multipli."
        )
    if metadata_columns:
        warnings.append(
            "Colonne non numeriche trattate come metadati genici: "
            + ", ".join(map(str, metadata_columns[:10]))
        )

    values = numeric[numeric_columns]
    non_numeric_cells = int(
        (data[numeric_columns].notna() & values.isna()).sum().sum()
    )
    if non_numeric_cells:
        warnings.append(
            f"{non_numeric_cells} celle non numeriche convertite in NaN nelle "
            f"colonne campione"
        )
    values.index = pd.Index(gene_ids, name=str(gene_column))
    gene_metadata = data[metadata_columns].copy()
    gene_metadata.index = values.index

    LOGGER.info(
        "Matrice letta: %d righe x %d campioni (colonna geni: '%s', separatore: %r)",
        values.shape[0],
        values.shape[1],
        gene_column,
        separator,
    )
    for message in warnings:
        LOGGER.warning(message)

    return ExpressionMatrix(
        values=values,
        gene_metadata=gene_metadata,
        data_type=profile.data_type,
        source_file=path.name,
        gene_column=str(gene_column),
        separator=separator,
        warnings=warnings,
    )


def _deduplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rename duplicate columns by adding a progressive suffix."""
    counts: dict[str, int] = {}
    new_columns: list[str] = []
    duplicated: list[str] = []
    for column in frame.columns:
        name = str(column).strip()
        if name in counts:
            counts[name] += 1
            duplicated.append(name)
            name = f"{name}.{counts[name]}"
        else:
            counts[name] = 0
        new_columns.append(name)
    frame.columns = new_columns
    if duplicated:
        return frame, "Colonne duplicate rinominate: " + ", ".join(sorted(set(duplicated)))
    return frame, ""


def log2_transform(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return ``log2(x + 1)`` (for descriptive exploration only)."""
    return np.log2(matrix.astype(float) + 1.0)
