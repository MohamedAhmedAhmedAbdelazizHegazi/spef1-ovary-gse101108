"""Export of tables, numbers for the manuscript and summaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

EXCEL_MAX_ROWS = 1_048_575


class ReportError(RuntimeError):
    """Output writing error."""


def _hint(path: Path, exc: Exception) -> str:
    return (
        f"Impossibile scrivere {path}: {exc}. Chiudere il file se aperto in Excel "
        f"e verificare i permessi della cartella."
    )


def save_workbook(sheets: Mapping[str, pd.DataFrame], path: Path) -> Path:
    """Save several tables to a single Excel file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                if frame is None:
                    continue
                data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
                if data.empty:
                    data = pd.DataFrame([{"info": "nessun risultato"}])
                if len(data) > EXCEL_MAX_ROWS:
                    LOGGER.warning(
                        "Foglio '%s': %d righe troncate a %d (usare il CSV completo)",
                        name, len(data), EXCEL_MAX_ROWS,
                    )
                    data = data.iloc[:EXCEL_MAX_ROWS]
                sheet = "".join(c for c in str(name) if c not in "[]:*?/\\")[:31]
                data.to_excel(writer, sheet_name=sheet, index=data.index.name is not None)
    except (OSError, PermissionError) as exc:
        raise ReportError(_hint(path, exc)) from exc
    LOGGER.info("Scritto %s (%d fogli)", path.name, len(sheets))
    return path


def save_csv(frame: pd.DataFrame, path: Path, index: bool = True) -> Path:
    """Save a table to CSV (compressed if the name ends in .gz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=index)
    except (OSError, PermissionError) as exc:
        raise ReportError(_hint(path, exc)) from exc
    LOGGER.info("Scritto %s (%d righe)", path.name, len(frame))
    return path


def save_json(payload: Any, path: Path) -> Path:
    """Save an object to JSON, converting the numpy types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)

    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)
    except (OSError, PermissionError) as exc:
        raise ReportError(_hint(path, exc)) from exc
    LOGGER.info("Scritto %s", path.name)
    return path


def save_text(lines: list[str], path: Path) -> Path:
    """Save a list of rows to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except (OSError, PermissionError) as exc:
        raise ReportError(_hint(path, exc)) from exc
    LOGGER.info("Scritto %s", path.name)
    return path


def build_manuscript_numbers(results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the numbers citable in the manuscript text.

    The file produced (``manuscript_numbers.json``) is the single source of the
    values reported in Results: no number is typed by hand into the text.

    """
    profile = results["profile"]
    detection = profile.detection_summary.set_index("histotype")
    numbers: dict[str, Any] = {
        "cohort": {
            "gse_id": results["gse_id"],
            "n_samples": int(results["n_samples"]),
            "n_genes_total": int(results["n_genes_total"]),
            "n_genes_expressed": int(results["n_genes_expressed"]),
            "histotype_counts": results["histotype_counts"],
            "median_library_size": float(results["median_library_size"]),
            "library_size_range": results["library_size_range"],
        },
        "detection": {
            str(index): {
                "n": int(row["n"]),
                "n_positive": int(row["n_positive"]),
                "percent": float(row["detection_percent"]),
                "median_counts": float(row["median_counts"]),
                "n_zero": int(row["n_zero"]),
            }
            for index, row in detection.iterrows()
        },
        "detection_pairwise_significant": profile.detection_pairwise.loc[
            profile.detection_pairwise["q_value_BH"] < 0.05,
            ["group_1", "group_2", "rate_1", "rate_2", "p_value", "q_value_BH"],
        ].to_dict("records"),
        "expression": {
            row["group"]: {
                "n": int(row["n"]),
                "median_log2": round(float(row["median"]), 3),
                "iqr": [round(float(row["q1"]), 3), round(float(row["q3"]), 3)],
            }
            for _, row in profile.expression_summary.iterrows()
        },
        "kruskal": profile.kruskal,
        "expression_pairwise": profile.expression_pairwise.to_dict("records"),
        "stage": profile.stage.to_dict("records"),
        "age": profile.age,
        "library_size_check": profile.library_size_check,
        "coexpression": results["coexpression_summary"],
        "panels": results["panel_summary"],
        "differential": results["differential_summary"],
        "enrichment": results["enrichment_summary"],
        "stratified": results.get("stratified_summary", []),
        "immune": results["immune_summary"],
        "notes": profile.notes,
    }
    return numbers


def summary_lines(numbers: Mapping[str, Any]) -> list[str]:
    """Readable text summary of the analysis."""
    cohort = numbers["cohort"]
    lines = [
        "=" * 78,
        f"ANALISI SPEF1 - {cohort['gse_id']}",
        "=" * 78,
        f"Campioni analizzati: {cohort['n_samples']}",
        f"Istotipi: " + ", ".join(f"{k}={v}" for k, v in cohort["histotype_counts"].items()),
        f"Geni totali: {cohort['n_genes_total']}  |  espressi (filtro): {cohort['n_genes_expressed']}",
        f"Library size mediana: {cohort['median_library_size']:.2e}",
        "",
        "RILEVABILITA' DEL BERSAGLIO",
    ]
    for histotype, values in numbers["detection"].items():
        lines.append(
            f"  {histotype:<22} {values['n_positive']:>3}/{values['n']:<3} "
            f"({values['percent']:>5.1f}%)  mediana={values['median_counts']:.1f} conteggi"
        )
    lines += [
        "",
        f"Kruskal-Wallis sui livelli: p = {numbers['kruskal']['p_value']:.4f}",
        f"Correlazione con l'eta': rho = {numbers['age']['rho']:.3f} "
        f"(p = {numbers['age']['p_value']:.3f})",
        f"Controllo library size: rho = {numbers['library_size_check']['rho']:.3f} "
        f"(p = {numbers['library_size_check']['p_value']:.3f})",
        "",
        "CO-ESPRESSIONE",
        f"  geni testati: {numbers['coexpression']['n_tested']}",
        f"  significativi (q<0.05): {numbers['coexpression']['n_significant']}",
        f"  con rho >= soglia: {numbers['coexpression']['n_above_rho']}",
        "",
        "ESPRESSIONE DIFFERENZIALE",
        f"  {numbers['differential']['comparison']}: "
        f"{numbers['differential']['n_significant']} geni con q<0.05",
        "",
        "ARRICCHIMENTO GO",
    ]
    for key, value in numbers["enrichment"].items():
        lines.append(f"  {key}: {value}")
    lines += ["", "IMMUNOLOGIA"]
    for key, value in numbers["immune"].items():
        lines.append(f"  {key}: {value}")
    lines.append("=" * 78)
    return lines
