"""Writing of the outputs (compressed CSV, multi-sheet Excel, JSON, text).

All functions explicitly handle the typical Windows errors (file open in
Excel, missing permissions) and the sheet size limits.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXCEL_MAX_ROWS = 1_048_575  # xlsx row limit minus the header
EXCEL_MAX_COLS = 16_384
SHEET_NAME_MAX = 31


class ExportError(RuntimeError):
    """Errore nella scrittura di un file di output."""


def _permission_hint(path: Path, exc: Exception) -> str:
    return (
        f"Impossibile scrivere {path}: {exc}. Chiudere il file se e' aperto in "
        f"Excel e verificare i permessi della cartella."
    )


def save_dataframe_csv(
    frame: pd.DataFrame, path: Path, index: bool = True, compression: str | None = "infer"
) -> Path:
    """Save a DataFrame to CSV (compressed if the name ends with ``.gz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=index, compression=compression)
    except (OSError, PermissionError) as exc:
        raise ExportError(_permission_hint(path, exc)) from exc
    LOGGER.info("Scritto %s (%d righe x %d colonne)", path.name, *frame.shape)
    return path


def _sanitize_sheet_name(name: str) -> str:
    """Make a name compatible with the Excel sheet constraints."""
    cleaned = "".join(ch for ch in str(name) if ch not in set("[]:*?/\\"))
    return (cleaned or "sheet")[:SHEET_NAME_MAX]


def _truncate_for_excel(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Truncate rows/columns exceeding the xlsx format limits."""
    result = frame
    if result.shape[0] > EXCEL_MAX_ROWS:
        LOGGER.warning(
            "'%s' ha %d righe: troncato a %d nel file Excel (usare il CSV per "
            "il dato completo).",
            label,
            result.shape[0],
            EXCEL_MAX_ROWS,
        )
        result = result.iloc[:EXCEL_MAX_ROWS]
    if result.shape[1] > EXCEL_MAX_COLS:
        LOGGER.warning(
            "'%s' ha %d colonne: troncato a %d nel file Excel.",
            label,
            result.shape[1],
            EXCEL_MAX_COLS,
        )
        result = result.iloc[:, :EXCEL_MAX_COLS]
    return result


def save_dataframe_excel(
    frame: pd.DataFrame, path: Path, sheet_name: str = "data", index: bool = False
) -> Path:
    """Save a DataFrame to a single-sheet Excel file."""
    return save_excel_workbook({sheet_name: frame}, path, index=index)


def save_excel_workbook(
    sheets: Mapping[str, pd.DataFrame], path: Path, index: bool = False
) -> Path:
    """Save several DataFrames to a single multi-sheet Excel file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                if frame is None:
                    continue
                data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
                data = _truncate_for_excel(data, name)
                data.to_excel(
                    writer, sheet_name=_sanitize_sheet_name(name), index=index
                )
    except (OSError, PermissionError) as exc:
        raise ExportError(_permission_hint(path, exc)) from exc
    LOGGER.info("Scritto %s (%d fogli)", path.name, len(sheets))
    return path


def save_json(payload: Any, path: Path) -> Path:
    """Save a serializable object to indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    except (OSError, PermissionError) as exc:
        raise ExportError(_permission_hint(path, exc)) from exc
    LOGGER.info("Scritto %s", path.name)
    return path


def save_text(lines: Sequence[str], path: Path) -> Path:
    """Save a list of rows to a UTF-8 text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(str(line) for line in lines), encoding="utf-8")
    except (OSError, PermissionError) as exc:
        raise ExportError(_permission_hint(path, exc)) from exc
    LOGGER.info("Scritto %s", path.name)
    return path


def build_sample_by_gene(
    gene_by_sample: pd.DataFrame, matching_table: pd.DataFrame
) -> pd.DataFrame:
    """Transform a gene x sample matrix into sample x gene.

    The matrix columns are relabeled with the GSM ID; the original label stays
    in the ``sample_id`` column.

    """
    transposed = gene_by_sample.transpose()
    transposed.index.name = "sample_id"
    transposed = transposed.reset_index()

    column_to_gsm = dict(
        zip(matching_table["matrix_column"], matching_table["gsm_id"])
    )
    transposed.insert(0, "gsm_id", transposed["sample_id"].map(column_to_gsm))
    return transposed


def merge_with_metadata(
    sample_by_gene: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join the sample x gene matrix with the available clinical metadata."""
    if metadata_columns is None:
        preferred = [
            "gsm_id",
            "title",
            "histotype_original",
            "histotype_normalized",
            "histotype_confidence",
            "needs_manual_review",
            "stage",
            "grade",
            "age",
            "sex",
            "overall_survival",
            "vital_status",
            "treatment",
            "tissue",
            "platform_id",
        ]
        metadata_columns = [c for c in preferred if c in metadata.columns]
    merged = sample_by_gene.merge(
        metadata[list(metadata_columns)], on="gsm_id", how="left", validate="one_to_one"
    )
    ordered = [c for c in metadata_columns if c in merged.columns]
    ordered += ["sample_id"] if "sample_id" in merged.columns else []
    rest = [c for c in merged.columns if c not in ordered]
    return merged[ordered + rest]
