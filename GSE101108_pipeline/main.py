"""GSE101108 pipeline: download, organization and preparation of the dataset.

Typical use::

    python main.py
    python main.py --genes ARID1A PIK3CA PTEN --histotypes "Clear cell" Mucinous
    python main.py --skip-download --no-mygene

"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config as config_module
from config import PipelineConfig, normalize_gene_list
from src import (
    downloader,
    expression_loader,
    exporters,
    filtering,
    gene_annotation,
    geo_metadata,
    quality_control,
    sample_matching,
)
from src.quality_control import RunRecorder

LOGGER = logging.getLogger("pipeline")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


# --------------------------------------------------------------------------- #
# CLI e logging                                                                #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Scarica e prepara il dataset GEO GSE101108 (carcinomi ovarici "
            "primitivi invasivi), escludendo gli istotipi sierosi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gse", default=None, help="Accession della serie GEO")
    parser.add_argument(
        "--genes", nargs="+", default=None, help="Lista dei geni di interesse"
    )
    parser.add_argument(
        "--histotypes",
        nargs="+",
        default=None,
        help='Istotipi ammessi, es. --histotypes "Clear cell" Endometrioid Mucinous',
    )
    parser.add_argument(
        "--exclude-histotypes",
        nargs="+",
        default=None,
        help="Istotipi da escludere sempre",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Scarica i file ed esce"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Usa i file gia' presenti in data/raw senza scaricare nulla",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Prosegue anche se il matching campioni/matrice e' sotto soglia",
    )
    parser.add_argument(
        "--keep-unspecified",
        action="store_true",
        help="Include i campioni 'Other or unspecified' nel dataset finale",
    )
    parser.add_argument(
        "--no-mygene",
        action="store_true",
        help="Disabilita la conversione degli ID genici tramite mygene.info",
    )
    parser.add_argument(
        "--no-ncbi-fallback",
        action="store_true",
        help=(
            "Disabilita l'annotazione di riserva basata su NCBI gene_info "
            "(usata quando mygene.info non e' raggiungibile)"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="File YAML di configurazione"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosita' del log su console",
    )
    return parser


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    """Configure logging to console and to file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level))
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    root.addHandler(console)

    logging.getLogger("GEOparse").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build the final configuration (defaults -> YAML -> CLI)."""
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
        cfg.project_root = Path(__file__).resolve().parent
    else:
        cfg = PipelineConfig()
    cfg.update(
        gse_id=args.gse,
        genes_of_interest=args.genes,
        allowed_histotypes=args.histotypes,
        excluded_histotypes=args.exclude_histotypes,
    )
    if args.force:
        cfg.force = True
    if args.keep_unspecified:
        cfg.keep_unspecified = True
    if args.no_mygene:
        cfg.use_mygene = False
    if args.no_ncbi_fallback:
        cfg.use_ncbi_fallback = False
    cfg.genes_of_interest = normalize_gene_list(cfg.genes_of_interest)
    return cfg


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #


def step_download(
    cfg: PipelineConfig, recorder: RunRecorder, skip: bool
) -> tuple[list[downloader.DownloadResult], dict[str, downloader.DownloadResult]]:
    """Download the supplementary files and the series metadata."""
    if skip:
        recorder.decide("download", "Download saltato (--skip-download)")
        return [], {}

    LOGGER.info("Directory supplementare: %s", cfg.suppl_url)
    suppl_results, remote_files = downloader.download_supplementary_files(
        cfg.suppl_url, cfg.raw_dir, timeout=cfg.download_timeout,
        retries=cfg.download_retries,
    )
    recorder.decide(
        "download",
        f"{len(remote_files)} file supplementari elencati, "
        f"{sum(1 for r in suppl_results if r.status in {'downloaded', 'cached'})} "
        f"disponibili localmente",
    )
    for result in suppl_results:
        if result.status == "failed":
            recorder.error("download", f"{result.file_name}: {result.message}")

    meta_results = downloader.download_geo_metadata_files(
        cfg.soft_url, cfg.matrix_url, cfg.raw_dir,
        timeout=cfg.download_timeout, retries=cfg.download_retries,
    )
    for name, result in meta_results.items():
        if result.status == "failed":
            recorder.error("download", f"{name}: {result.message}")
    return suppl_results, meta_results


def step_metadata(
    cfg: PipelineConfig, recorder: RunRecorder
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and normalize the GEO metadata."""
    soft_path = cfg.raw_dir / f"{cfg.gse_id}_family.soft.gz"
    series = geo_metadata.load_series(soft_path, cfg.gse_id)
    raw, normalized = geo_metadata.build_metadata_frames(series)

    missing = geo_metadata.missing_clinical_columns(normalized)
    if missing:
        recorder.warn(
            "metadati",
            "Variabili cliniche NON disponibili in GEO per questa serie: "
            + ", ".join(missing),
        )
    recorder.decide(
        "metadati",
        f"{len(normalized)} campioni GSM letti; colonne cliniche disponibili: "
        + ", ".join(
            c for c in geo_metadata.EXPECTED_CLINICAL_COLUMNS if c in normalized.columns
        ),
    )
    exporters.save_dataframe_excel(
        raw, cfg.out(cfg.metadata_dir, "metadata_raw.xlsx"), sheet_name="metadata_raw"
    )
    exporters.save_dataframe_excel(
        normalized,
        cfg.out(cfg.metadata_dir, "metadata_normalized.xlsx"),
        sheet_name="metadata",
    )
    return raw, normalized


def step_histotypes(
    cfg: PipelineConfig, normalized: pd.DataFrame, recorder: RunRecorder
) -> tuple[pd.DataFrame, filtering.SampleSets, pd.DataFrame]:
    """Classify the histotypes and build the sample sets."""
    column = geo_metadata.find_histotype_column(
        normalized, config_module.HISTOTYPE_FIELD_KEYWORDS
    )
    if column is None:
        recorder.error(
            "istotipo",
            "Colonna dell'istotipo non identificata: nessun filtro per istotipo "
            "potra' essere applicato.",
        )
    else:
        recorder.decide("istotipo", f"Colonna dell'istotipo individuata: '{column}'")

    classified = filtering.classify_histotypes(normalized, column)
    classification = filtering.histotype_classification_table(classified)
    summary = filtering.histotype_summary(classified)

    sets = filtering.build_sample_sets(
        classified,
        cfg.allowed_histotypes,
        cfg.excluded_histotypes,
        keep_unspecified=cfg.keep_unspecified,
    )
    recorder.decide(
        "istotipo",
        "Conteggi per istotipo: "
        + "; ".join(f"{r.histotype}={r.n_samples}" for r in summary.itertuples()),
    )
    if not sets.to_review.empty:
        recorder.warn(
            "istotipo",
            f"{len(sets.to_review)} campioni da revisionare manualmente "
            f"(istotipo dubbio o mancante)",
        )

    exporters.save_excel_workbook(
        {"classificazione": classification, "conteggi": summary},
        cfg.out(cfg.metadata_dir, "histotype_classification.xlsx"),
    )
    exporters.save_dataframe_excel(
        sets.allowed, cfg.out(cfg.metadata_dir, "allowed_samples.xlsx"),
        sheet_name="allowed",
    )
    exporters.save_dataframe_excel(
        sets.to_review, cfg.out(cfg.metadata_dir, "samples_to_review.xlsx"),
        sheet_name="to_review",
    )
    return classified, sets, summary


def step_expression(
    cfg: PipelineConfig, recorder: RunRecorder
) -> tuple[expression_loader.ExpressionMatrix, pd.DataFrame]:
    """Locate and read the expression matrix."""
    profiles = expression_loader.profile_directory(
        cfg.raw_dir, config_module.GENE_ID_COLUMN_CANDIDATES
    )
    best = expression_loader.select_best_candidate(profiles)
    table = expression_loader.candidates_table(profiles, best.file_name)
    recorder.decide(
        "matrice",
        f"Selezionato '{best.file_name}' (tipo={best.data_type}, "
        f"score={best.score:.1f}) fra {len(profiles)} file esaminati: {best.reason}",
    )
    matrix = expression_loader.read_expression_matrix(
        best, config_module.GENE_ID_COLUMN_CANDIDATES
    )
    for message in matrix.warnings:
        recorder.warn("matrice", message)
    return matrix, table


def step_annotation(
    cfg: PipelineConfig,
    matrix: expression_loader.ExpressionMatrix,
    recorder: RunRecorder,
) -> tuple[pd.DataFrame, pd.DataFrame, gene_annotation.AggregationResult]:
    """Annota gli identificativi genici e aggrega i duplicati."""
    annotation = gene_annotation.build_annotation_table(
        matrix.values.index.tolist(),
        cfg.cache_dir,
        use_mygene=cfg.use_mygene,
        use_ncbi_fallback=cfg.use_ncbi_fallback,
        manual_map_path=cfg.manual_gene_map,
    )
    manual = int((annotation["conversion_status"] == "manual_map").sum())
    if manual:
        recorder.decide(
            "annotazione",
            f"{manual} identificativi risolti tramite la mappa manuale "
            f"{cfg.manual_gene_map.name} (ID Ensembl ritirati, annotazione hg19)",
        )
    symbol_map = dict(zip(annotation["original_id"], annotation["gene_symbol"]))
    symbols = [symbol_map.get(str(i), str(i).upper()) for i in matrix.values.index]

    aggregation = gene_annotation.aggregate_duplicate_genes(
        matrix.values, symbols, is_count_like=matrix.is_count_like
    )
    recorder.decide(
        "annotazione",
        f"Righe duplicate aggregate con metodo '{aggregation.method}' "
        f"({'somma per i conteggi' if aggregation.method == 'sum' else 'media per TPM/FPKM'}): "
        f"{aggregation.n_rows_before} -> {aggregation.n_rows_after} righe, "
        f"{aggregation.n_duplicated_symbols} simboli duplicati",
    )
    not_converted = int((annotation["conversion_status"] == "not_found").sum())
    if not_converted:
        recorder.warn(
            "annotazione",
            f"{not_converted} identificativi non convertiti in simbolo genico "
            f"(mantenuto l'ID originale)",
        )
    exporters.save_dataframe_excel(
        annotation, cfg.out(cfg.processed_dir, "gene_annotation.xlsx"),
        sheet_name="annotation",
    )
    return annotation, aggregation.matrix, aggregation


def step_matching(
    cfg: PipelineConfig,
    classified: pd.DataFrame,
    sets: filtering.SampleSets,
    symbol_matrix: pd.DataFrame,
    recorder: RunRecorder,
) -> tuple[sample_matching.MatchingResult, bool, float]:
    """Match the matrix columns to the GEO samples."""
    matching = sample_matching.match_samples_to_columns(
        classified, symbol_matrix.columns.tolist()
    )
    exporters.save_excel_workbook(
        {
            "matching": matching.table,
            "conflitti": matching.conflicts,
            "colonne_non_associate": quality_control.sequence_to_frame(
                matching.unmatched_columns, "matrix_column"
            ),
            "campioni_non_associati": quality_control.sequence_to_frame(
                matching.unmatched_samples, "gsm_id"
            ),
        },
        cfg.out(cfg.processed_dir, "sample_matching.xlsx"),
    )
    ok, fraction, message = sample_matching.check_match_coverage(
        matching, sets.allowed["gsm_id"], cfg.min_match_fraction, force=cfg.force
    )
    if fraction < cfg.min_match_fraction:
        recorder.error("matching", message)
    else:
        recorder.decide("matching", message)
    if not matching.conflicts.empty:
        recorder.warn(
            "matching",
            f"{len(matching.conflicts)} campioni con corrispondenze ambigue: "
            f"vedere il foglio 'conflitti'",
        )
    return matching, ok, fraction


def _rename_columns_to_gsm(
    matrix: pd.DataFrame, matching: sample_matching.MatchingResult
) -> pd.DataFrame:
    """Rename the columns as ``GSM|original label`` where possible."""
    reverse = {column: gsm for gsm, column in matching.matched.items()}
    renamed = matrix.rename(
        columns={c: f"{reverse[c]}|{c}" for c in matrix.columns if c in reverse}
    )
    return renamed


def step_datasets(
    cfg: PipelineConfig,
    symbol_matrix: pd.DataFrame,
    matrix_info: expression_loader.ExpressionMatrix,
    sets: filtering.SampleSets,
    matching: sample_matching.MatchingResult,
    recorder: RunRecorder,
) -> dict[str, pd.DataFrame]:
    """Save the complete, non-serous and allowed-histotype matrices."""
    datasets: dict[str, pd.DataFrame] = {}

    def columns_for(frame: pd.DataFrame) -> list[str]:
        return [matching.matched[g] for g in frame["gsm_id"] if g in matching.matched]

    definitions = {
        "complete": (symbol_matrix.columns.tolist(), "complete_expression_matrix"),
        "non_serous": (columns_for(sets.non_serous), "non_serous_expression_matrix"),
        "allowed": (
            columns_for(sets.allowed),
            "allowed_histotypes_expression_matrix",
        ),
    }
    for key, (columns, stem) in definitions.items():
        subset = filtering.filter_matrix_by_samples(symbol_matrix, columns)
        datasets[key] = subset
        renamed = _rename_columns_to_gsm(subset, matching)
        suffix = f"{stem}_{matrix_info.data_type}.csv.gz"
        exporters.save_dataframe_csv(renamed, cfg.out(cfg.processed_dir, suffix))
        # nome "canonico" richiesto dalle specifiche (stesso contenuto)
        exporters.save_dataframe_csv(renamed, cfg.out(cfg.processed_dir, f"{stem}.csv.gz"))
        if subset.empty:
            recorder.warn("dataset", f"Il dataset '{key}' non contiene campioni")

    allowed_log2 = expression_loader.log2_transform(datasets["allowed"])
    exporters.save_dataframe_csv(
        _rename_columns_to_gsm(allowed_log2, matching),
        cfg.out(cfg.processed_dir, "allowed_histotypes_log2_matrix.csv.gz"),
    )
    recorder.decide(
        "dataset",
        f"Matrici salvate come '{matrix_info.data_type}' piu' la versione "
        f"log2(x+1) per le analisi descrittive (non usare log2/TPM con DESeq2)",
    )
    return datasets


def step_genes(
    cfg: PipelineConfig,
    allowed_matrix: pd.DataFrame,
    annotation: pd.DataFrame,
    classified: pd.DataFrame,
    matching: sample_matching.MatchingResult,
    matrix_info: expression_loader.ExpressionMatrix,
    recorder: RunRecorder,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract the genes of interest and produce the final files."""
    selected, lookup = gene_annotation.extract_genes_of_interest(
        allowed_matrix, cfg.genes_of_interest, annotation
    )
    found = lookup.loc[lookup["found"], "gene"].tolist()
    missing = lookup.loc[~lookup["found"], "gene"].tolist()
    recorder.decide(
        "geni",
        f"Geni trovati {len(found)}/{len(cfg.genes_of_interest)}: {', '.join(found)}",
    )
    if missing:
        recorder.warn("geni", "Geni NON trovati: " + ", ".join(missing))

    selected_log2 = expression_loader.log2_transform(selected)
    exporters.save_excel_workbook(
        {
            f"{matrix_info.data_type}": selected.reset_index(),
            "log2": selected_log2.round(4).reset_index(),
            "ricerca_geni": lookup,
        },
        cfg.out(cfg.processed_dir, "selected_genes_gene_by_sample.xlsx"),
    )

    sample_by_gene = exporters.build_sample_by_gene(selected, matching.table)
    sample_by_gene_log2 = exporters.build_sample_by_gene(selected_log2, matching.table)
    exporters.save_excel_workbook(
        {
            f"{matrix_info.data_type}": sample_by_gene,
            "log2": sample_by_gene_log2.round(4),
        },
        cfg.out(cfg.processed_dir, "selected_genes_sample_by_gene.xlsx"),
    )

    merged = exporters.merge_with_metadata(sample_by_gene, classified)
    merged_log2 = exporters.merge_with_metadata(sample_by_gene_log2, classified)
    exporters.save_excel_workbook(
        {
            f"dati_{matrix_info.data_type}": merged,
            "dati_log2": merged_log2.round(4),
            "ricerca_geni": lookup,
        },
        cfg.out(cfg.processed_dir, "selected_genes_with_metadata.xlsx"),
    )
    exporters.save_dataframe_excel(
        lookup.loc[~lookup["found"]],
        cfg.out(cfg.processed_dir, "genes_not_found.xlsx"),
        sheet_name="not_found",
    )
    return selected, lookup, merged


def step_quality_control(
    cfg: PipelineConfig,
    matrix_info: expression_loader.ExpressionMatrix,
    symbol_matrix: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    lookup: pd.DataFrame,
    matching: sample_matching.MatchingResult,
    classified: pd.DataFrame,
    histotype_summary: pd.DataFrame,
    annotation: pd.DataFrame,
    candidates: pd.DataFrame,
    downloads: list[downloader.DownloadResult],
    summary: dict[str, Any],
    recorder: RunRecorder,
) -> None:
    """Compute the QC metrics, generate the figures and write the report."""
    overview = quality_control.matrix_overview(symbol_matrix, matrix_info.data_type)
    library = quality_control.library_size_table(symbol_matrix)
    log_all = expression_loader.log2_transform(symbol_matrix)
    distribution = quality_control.expression_distribution_table(log_all)
    duplicates = quality_control.duplicated_report(symbol_matrix)
    missing_values = quality_control.missing_values_table(symbol_matrix)

    groups = quality_control.sample_group_map(matching.table)
    labels = quality_control.sample_label_map(matching.table)

    quality_control.plot_library_sizes(
        library, cfg.figures_dir / f"{cfg.gse_id}_library_size.png", cfg.figure_dpi
    )
    quality_control.plot_expression_boxplot(
        log_all, cfg.figures_dir / f"{cfg.gse_id}_expression_distribution.png",
        cfg.figure_dpi,
    )
    allowed_log2 = expression_loader.log2_transform(datasets["allowed"])
    selected_log2 = expression_loader.log2_transform(selected)
    quality_control.plot_selected_genes_heatmap(
        selected_log2, labels,
        cfg.figures_dir / f"{cfg.gse_id}_selected_genes_heatmap.png", cfg.figure_dpi,
    )
    _, pca_coords = quality_control.plot_pca(
        allowed_log2, groups, cfg.figures_dir / f"{cfg.gse_id}_pca_samples.png",
        cfg.figure_dpi,
    )
    quality_control.plot_genes_by_histotype(
        selected_log2, groups,
        cfg.figures_dir / f"{cfg.gse_id}_selected_genes_by_histotype.png",
        cfg.figure_dpi,
    )

    sheets = {
        "riepilogo": quality_control.build_summary_table(summary),
        "download": pd.DataFrame([d.to_dict() for d in downloads])
        if downloads
        else pd.DataFrame([{"info": "download saltato"}]),
        "file_candidati": candidates,
        "campioni": classified.drop(columns=["metadata_full_text"], errors="ignore"),
        "istotipi": histotype_summary,
        "matching": matching.table,
        "annotazione_genica": annotation.head(50000),
        "geni_selezionati": lookup,
        "matrice_qc": overview,
        "library_size": library,
        "distribuzione": distribution,
        "duplicati": duplicates,
        "valori_mancanti": missing_values,
    }
    if pca_coords is not None:
        sheets["pca"] = pca_coords
    sheets.update(recorder.as_frames())
    exporters.save_excel_workbook(
        sheets, cfg.reports_dir / f"{cfg.gse_id}_quality_control_report.xlsx"
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    """Run the full pipeline. Return the exit code."""
    started = time.time()
    cfg = build_config(args)
    cfg.create_directories()
    setup_logging(cfg.log_file, args.log_level)
    recorder = RunRecorder()

    LOGGER.info("=" * 70)
    LOGGER.info("Pipeline %s avviata (%s)", cfg.gse_id, datetime.now(timezone.utc).isoformat())
    LOGGER.info("Python %s su %s", platform.python_version(), platform.platform())
    LOGGER.info("Configurazione: %s", json.dumps(cfg.as_dict(), ensure_ascii=False))
    LOGGER.info("=" * 70)

    downloads, _ = step_download(cfg, recorder, skip=args.skip_download)
    if args.download_only:
        LOGGER.info("Modalita' --download-only: pipeline terminata dopo il download.")
        return 0

    _, normalized = step_metadata(cfg, recorder)
    classified, sets, histotype_summary = step_histotypes(cfg, normalized, recorder)
    matrix_info, candidates = step_expression(cfg, recorder)
    annotation, symbol_matrix, aggregation = step_annotation(cfg, matrix_info, recorder)
    matching, coverage_ok, fraction = step_matching(
        cfg, classified, sets, symbol_matrix, recorder
    )

    summary: dict[str, Any] = {
        "gse_id": cfg.gse_id,
        "data_esecuzione": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_matrice": matrix_info.source_file,
        "tipo_di_dato": matrix_info.data_type,
        "colonna_geni": matrix_info.gene_column,
        "n_campioni_geo": len(classified),
        "n_campioni_matrice": symbol_matrix.shape[1],
        "n_geni_matrice": symbol_matrix.shape[0],
        "metodo_aggregazione_duplicati": aggregation.method,
        "n_simboli_duplicati": aggregation.n_duplicated_symbols,
        "istotipi_ammessi": cfg.allowed_histotypes,
        "istotipi_esclusi": cfg.excluded_histotypes,
        "conteggi_istotipo": dict(
            zip(histotype_summary["histotype"], histotype_summary["n_samples"])
        ),
        **sets.summary(),
        "campioni_associati": len(matching.matched),
        "frazione_campioni_ammessi_associati": round(fraction, 4),
        "geni_richiesti": cfg.genes_of_interest,
    }

    if not coverage_ok:
        recorder.error(
            "matching",
            "Copertura del matching insufficiente: il dataset finale NON e' stato "
            "creato. Rilanciare con --force per forzare la creazione.",
        )
        summary["esito"] = "interrotto: matching insufficiente"
        _finalize(cfg, summary, recorder, started)
        return 2

    datasets = step_datasets(cfg, symbol_matrix, matrix_info, sets, matching, recorder)
    selected, lookup, merged = step_genes(
        cfg, datasets["allowed"], annotation, classified, matching, matrix_info, recorder
    )
    summary["n_campioni_dataset_finale"] = len(merged)
    summary["geni_trovati"] = lookup.loc[lookup["found"], "gene"].tolist()
    summary["geni_non_trovati"] = lookup.loc[~lookup["found"], "gene"].tolist()
    summary["esito"] = "completato"

    step_quality_control(
        cfg, matrix_info, symbol_matrix, datasets, selected, lookup, matching,
        classified, histotype_summary, annotation, candidates, downloads, summary,
        recorder,
    )
    _finalize(cfg, summary, recorder, started)
    return 0


def _finalize(
    cfg: PipelineConfig, summary: dict[str, Any], recorder: RunRecorder, started: float
) -> None:
    """Write the text and JSON summary and close the log."""
    summary["durata_secondi"] = round(time.time() - started, 1)
    summary["n_errori"] = len(recorder.errors)
    summary["n_warning"] = len(recorder.warnings)

    lines = quality_control.summary_text_lines(summary)
    lines.append("")
    lines.append("DECISIONI AUTOMATICHE")
    lines += [f"  - [{d['step']}] {d['message']}" for d in recorder.decisions]
    if recorder.warnings:
        lines += ["", "WARNING"] + [
            f"  - [{w['step']}] {w['message']}" for w in recorder.warnings
        ]
    if recorder.errors:
        lines += ["", "ERRORI"] + [
            f"  - [{e['step']}] {e['message']}" for e in recorder.errors
        ]

    exporters.save_text(lines, cfg.reports_dir / f"{cfg.gse_id}_processing_summary.txt")
    exporters.save_json(
        {
            "summary": summary,
            "config": cfg.as_dict(),
            "decisions": recorder.decisions,
            "warnings": recorder.warnings,
            "errors": recorder.errors,
        },
        cfg.reports_dir / f"{cfg.gse_id}_processing_summary.json",
    )
    LOGGER.info(
        "Pipeline terminata in %.1f s (%d errori, %d warning). Report in %s",
        summary["durata_secondi"],
        len(recorder.errors),
        len(recorder.warnings),
        cfg.reports_dir,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point with top-level error handling."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
        return 130
    except downloader.DownloadError as exc:
        logging.getLogger("pipeline").exception("Errore di download")
        print(f"\nERRORE DI DOWNLOAD: {exc}", file=sys.stderr)
        return 3
    except geo_metadata.MetadataError as exc:
        logging.getLogger("pipeline").exception("Errore sui metadati")
        print(f"\nERRORE SUI METADATI GEO: {exc}", file=sys.stderr)
        return 4
    except expression_loader.ExpressionLoadError as exc:
        logging.getLogger("pipeline").exception("Errore sulla matrice")
        print(f"\nERRORE SULLA MATRICE DI ESPRESSIONE: {exc}", file=sys.stderr)
        return 5
    except exporters.ExportError as exc:
        logging.getLogger("pipeline").exception("Errore di scrittura")
        print(f"\nERRORE DI SCRITTURA: {exc}", file=sys.stderr)
        return 6
    except Exception as exc:  # rete assente, bug, formati imprevisti
        logging.getLogger("pipeline").exception("Errore non gestito")
        print(
            f"\nERRORE INATTESO: {exc}\nDettagli completi nel file di log "
            f"(logs/*.log).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
