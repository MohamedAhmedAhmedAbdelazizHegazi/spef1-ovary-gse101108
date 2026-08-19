"""Parsing of the GEO metadata (series and GSM samples).

The series is read from a local SOFT file already downloaded (GEOparse) with a
fallback on the series matrix, so as not to depend on GEOparse's internal
downloader.

"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Synonyms -> canonical column name for the clinical characteristics.
CHARACTERISTIC_ALIASES: dict[str, str] = {
    "histotype": "histotype",
    "histology": "histotype",
    "histological type": "histotype",
    "histological subtype": "histotype",
    "histologic type": "histotype",
    "tumor type": "histotype",
    "tumour type": "histotype",
    "subtype": "histotype",
    "diagnosis": "histotype",
    "figo stage": "stage",
    "stage": "stage",
    "tumor stage": "stage",
    "grade": "grade",
    "tumor grade": "grade",
    "differentiation": "grade",
    "age": "age",
    "age at diagnosis": "age",
    "sex": "sex",
    "gender": "sex",
    "os": "overall_survival",
    "overall survival": "overall_survival",
    "survival": "overall_survival",
    "survival time": "overall_survival",
    "follow up": "overall_survival",
    "vital status": "vital_status",
    "status": "vital_status",
    "dead": "vital_status",
    "treatment": "treatment",
    "therapy": "treatment",
    "chemotherapy": "treatment",
    "tissue": "tissue",
}

#: Top-level fields extracted from each GSM.
SIMPLE_FIELDS: dict[str, str] = {
    "title": "title",
    "source_name_ch1": "source_name",
    "organism_ch1": "organism",
    "molecule_ch1": "molecule",
    "library_strategy": "library_strategy",
    "library_source": "library_source",
    "library_selection": "library_selection",
    "platform_id": "platform_id",
    "instrument_model": "instrument_model",
    "type": "sample_type",
    "description": "description",
    "status": "status",
    "submission_date": "submission_date",
    "last_update_date": "last_update_date",
    "relation": "relation",
    "data_processing": "data_processing",
}

#: Expected clinical columns: if absent in the dataset a warning is emitted.
EXPECTED_CLINICAL_COLUMNS = (
    "histotype",
    "stage",
    "grade",
    "age",
    "sex",
    "overall_survival",
    "vital_status",
    "treatment",
)


class MetadataError(RuntimeError):
    """Error in the retrieval or parsing of the GEO metadata."""


# --------------------------------------------------------------------------- #
# Reading of the series
# --------------------------------------------------------------------------- #


def load_series(soft_path: Path, gse_id: str) -> Any:
    """Load a GEOparse GSE object from a local SOFT file.

    Args:
        soft_path: path of the ``*_family.soft.gz`` file already downloaded.
        gse_id: series accession (used only in error messages).

    Raises:
        MetadataError: if the file does not exist or GEOparse cannot read it.

    """
    soft_path = Path(soft_path)
    if not soft_path.is_file():
        raise MetadataError(
            f"File SOFT non trovato per {gse_id}: {soft_path}. "
            f"Eseguire la pipeline senza --skip-download."
        )
    try:
        import GEOparse  # local import: the library is slow to import

        series = GEOparse.get_GEO(filepath=str(soft_path), geotype="GSE", silent=True)
    except Exception as exc:  # GEOparse solleva eccezioni molto varie
        raise MetadataError(
            f"GEOparse non e' riuscito a leggere {soft_path}: {exc}. "
            f"Cancellare il file e rilanciare la pipeline per riscaricarlo."
        ) from exc

    if not getattr(series, "gsms", None):
        raise MetadataError(
            f"Il file SOFT {soft_path} non contiene campioni GSM: "
            f"la serie {gse_id} potrebbe non essere ancora pubblica."
        )
    LOGGER.info("Serie %s caricata: %d campioni GSM", gse_id, len(series.gsms))
    return series


# --------------------------------------------------------------------------- #
# Parsing of the characteristics
# --------------------------------------------------------------------------- #


def normalize_key(key: str) -> str:
    """Normalize a ``characteristics_ch1`` key.

    Lower case, separators (``-``, ``_``, ``.``) converted to spaces, multiple
    spaces collapsed.

    """
    key = str(key).strip().lower()
    key = re.sub(r"[\-_\.]+", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip()


def canonical_key(key: str) -> str:
    """Map a normalized key onto the canonical column name."""
    norm = normalize_key(key)
    if norm in CHARACTERISTIC_ALIASES:
        return CHARACTERISTIC_ALIASES[norm]
    for alias, canonical in CHARACTERISTIC_ALIASES.items():
        if norm == alias or norm.startswith(alias + " ") or norm.endswith(" " + alias):
            return canonical
    return norm.replace(" ", "_")


def parse_characteristics(entries: Iterable[str]) -> dict[str, str]:
    """Convert ``["histotype: clear cell", ...]`` into a dictionary.

    Handles duplicate keys (values are concatenated with ``" | "``), entries
    without a separator, spaces and letter case.

    """
    parsed: dict[str, list[str]] = {}
    for index, raw in enumerate(entries or []):
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            key, _, value = text.partition(":")
        else:
            key, value = f"characteristic_{index + 1}", text
        column = canonical_key(key)
        value = value.strip()
        if not column:
            continue
        parsed.setdefault(column, [])
        if value:
            parsed[column].append(value)
    return {key: " | ".join(values) for key, values in parsed.items() if values}


def _flatten(values: Any, separator: str = " | ") -> str:
    """Reduce a list of GEOparse metadata values to a string."""
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return separator.join(str(v).strip() for v in values if str(v).strip())
    return str(values).strip()


def _raw_text(metadata: Mapping[str, Any]) -> str:
    """Serialize the whole block of original sample metadata."""
    return "\n".join(
        f"!{key} = {_flatten(value, ' ;; ')}" for key, value in sorted(metadata.items())
    )


# --------------------------------------------------------------------------- #
# Building of the tables
# --------------------------------------------------------------------------- #


def build_metadata_frames(series: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the raw and normalized metadata tables.

    Returns:
        ``(raw, normalized)``: the first holds one field per original GEO key, the
        second the structured clinical columns.

    """
    raw_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []

    for gsm_id, gsm in series.gsms.items():
        metadata: Mapping[str, Any] = gsm.metadata

        raw_row: dict[str, Any] = {"gsm_id": gsm_id}
        for key, value in metadata.items():
            raw_row[key] = _flatten(value)
        raw_rows.append(raw_row)

        norm_row: dict[str, Any] = {"gsm_id": gsm_id}
        for geo_key, column in SIMPLE_FIELDS.items():
            norm_row[column] = _flatten(metadata.get(geo_key))
        characteristics = parse_characteristics(metadata.get("characteristics_ch1", []))
        norm_row.update(characteristics)
        norm_row["characteristics_raw"] = _flatten(
            metadata.get("characteristics_ch1", []), " ;; "
        )
        norm_row["metadata_full_text"] = _raw_text(metadata)
        norm_rows.append(norm_row)

    raw = pd.DataFrame(raw_rows).convert_dtypes()
    normalized = pd.DataFrame(norm_rows)

    normalized = _coerce_numeric(normalized, ("age", "overall_survival"))
    normalized = _reorder_columns(normalized)
    _report_missing_clinical_columns(normalized)
    return raw, normalized


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert the indicated columns to numeric, if present."""
    for column in columns:
        if column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            failed = int(converted.isna().sum() - frame[column].isna().sum())
            if failed > 0:
                LOGGER.warning(
                    "Colonna '%s': %d valori non numerici impostati a NA", column, failed
                )
            frame[column] = converted
    return frame


def _reorder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Move the most informative columns to the front."""
    preferred = [
        "gsm_id",
        "title",
        "source_name",
        "histotype",
        "stage",
        "grade",
        "age",
        "sex",
        "overall_survival",
        "vital_status",
        "treatment",
        "tissue",
        "organism",
        "molecule",
        "library_strategy",
        "platform_id",
    ]
    ordered = [c for c in preferred if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    return frame[ordered]


def _report_missing_clinical_columns(frame: pd.DataFrame) -> list[str]:
    """Record which expected clinical variables are missing."""
    missing = [c for c in EXPECTED_CLINICAL_COLUMNS if c not in frame.columns]
    for column in missing:
        LOGGER.warning(
            "Variabile clinica '%s' NON disponibile nei metadati GEO: "
            "le analisi che la richiedono non saranno possibili.",
            column,
        )
    present = [c for c in EXPECTED_CLINICAL_COLUMNS if c in frame.columns]
    LOGGER.info("Variabili cliniche disponibili: %s", ", ".join(present) or "nessuna")
    return missing


def missing_clinical_columns(frame: pd.DataFrame) -> list[str]:
    """List the expected but missing clinical variables."""
    return [c for c in EXPECTED_CLINICAL_COLUMNS if c not in frame.columns]


def find_histotype_column(frame: pd.DataFrame, keywords: Iterable[str]) -> str | None:
    """Locate the column that holds the histotype.

    The search is based on the terms configured in
    ``config.HISTOTYPE_FIELD_KEYWORDS`` and ignores case, spaces and separators.

    """
    normalized = {normalize_key(col): col for col in frame.columns}
    keywords = [normalize_key(k) for k in keywords]

    for keyword in keywords:  # corrispondenza esatta
        if keyword in normalized:
            LOGGER.info("Colonna istotipo individuata (esatta): %s", normalized[keyword])
            return normalized[keyword]
    for keyword in keywords:  # corrispondenza parziale
        for norm_col, original in normalized.items():
            if keyword in norm_col:
                LOGGER.info(
                    "Colonna istotipo individuata (parziale, '%s'): %s",
                    keyword,
                    original,
                )
                return original
    LOGGER.error(
        "Nessuna colonna dell'istotipo individuata tra: %s", list(frame.columns)
    )
    return None


def read_series_matrix_header(matrix_path: Path) -> pd.DataFrame:
    """Read the ``!Sample_*`` lines of a series matrix as a table.

    Used as a fallback/check when the SOFT file is not available.

    """
    matrix_path = Path(matrix_path)
    opener = gzip.open if matrix_path.suffix == ".gz" else open
    rows: dict[str, list[str]] = {}
    with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key = parts[0].lstrip("!")
            values = parts[1:]
            if key in rows:
                index = 2
                while f"{key}_{index}" in rows:
                    index += 1
                key = f"{key}_{index}"
            rows[key] = values
    if not rows:
        raise MetadataError(f"Nessuna riga !Sample_ trovata in {matrix_path}")
    return pd.DataFrame(rows)
