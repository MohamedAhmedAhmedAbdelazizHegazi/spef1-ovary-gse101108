"""SPEF1 bioinformatic analysis in the GSE101108 cohort.

Use::

    python main.py
    python main.py --gene SPEF1 --detection-threshold 5
    python main.py --skip-enrichment      # without a connection to NCBI

"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config as config_module
from config import AnalysisConfig
from src import analyses, data_loading, enrichment, figures, reporting
from src.statistics import format_p

LOGGER = logging.getLogger("spef1")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Analisi bioinformatica del gene bersaglio (SPEF1) nella coorte "
            "GSE101108 di carcinomi ovarici primitivi invasivi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gene", default=None, help="Gene bersaglio")
    parser.add_argument(
        "--detection-threshold", type=int, default=None,
        help="Conteggi grezzi minimi per definire un campione positivo",
    )
    parser.add_argument(
        "--min-counts", type=int, default=None, help="Filtro di espressione: conteggi minimi"
    )
    parser.add_argument(
        "--min-fraction", type=float, default=None,
        help="Filtro di espressione: frazione minima di campioni",
    )
    parser.add_argument(
        "--pipeline-root", type=Path, default=None,
        help="Cartella della pipeline GSE101108 (input)",
    )
    parser.add_argument(
        "--include-unspecified", action="store_true",
        help="Include i campioni con istotipo non classificabile",
    )
    parser.add_argument(
        "--skip-enrichment", action="store_true",
        help="Salta l'arricchimento GO (utile senza connessione)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosita' del log su console",
    )
    return parser


def setup_logging(log_file: Path, level: str) -> None:
    """Configure logging to console and to file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    console.setLevel(getattr(logging, level))
    root.addHandler(console)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    """Build the configuration from the defaults and the CLI."""
    payload: dict[str, Any] = {
        "target_gene": args.gene,
        "detection_threshold": args.detection_threshold,
        "min_counts": args.min_counts,
        "min_fraction": args.min_fraction,
        "pipeline_root": args.pipeline_root,
    }
    cfg = AnalysisConfig.from_mapping({k: v for k, v in payload.items() if v is not None})
    cfg.include_unspecified = args.include_unspecified
    cfg.offline = args.skip_enrichment
    return cfg


# --------------------------------------------------------------------------- #
# Pipeline di analisi                                                          #
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    """Run the full analysis and return the exit code."""
    started = time.time()
    cfg = build_config(args)
    cfg.create_directories()
    setup_logging(cfg.log_file, args.log_level)

    LOGGER.info("=" * 70)
    LOGGER.info("Analisi %s su %s (%s)", cfg.target_gene, cfg.gse_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"))
    LOGGER.info("Python %s su %s", platform.python_version(), platform.platform())
    LOGGER.info("=" * 70)

    # ------------------------------------------------------------------ dati #
    dataset = data_loading.load_dataset(
        cfg.counts_file,
        cfg.metadata_file,
        cfg.histotype_file,
        config_module.HISTOTYPE_ORDER,
        include_unspecified=cfg.include_unspecified,
    )
    expressed = data_loading.filter_expressed_genes(dataset, cfg.min_counts, cfg.min_fraction)
    raw, _, log_values = data_loading.target_series(dataset, cfg.target_gene)

    # -------------------------------------------------------------- bersaglio #
    profile = analyses.profile_target(
        dataset, raw, log_values,
        cfg.detection_threshold,
        config_module.DETECTION_THRESHOLDS_SENSITIVITY,
        config_module.HISTOTYPE_ORDER,
    )
    detection_group, median_group = analyses.split_by_target(
        raw, log_values, cfg.detection_threshold
    )

    # ------------------------------------------------------------ co-espressione #
    coexpression = analyses.coexpression_analysis(
        dataset, expressed, log_values, cfg.target_gene
    )
    panels = pd.concat(
        [
            analyses.panel_correlations(
                dataset, log_values, config_module.MANUSCRIPT_GEPIA_GENES,
                "GEPIA2 top-10 (TCGA)",
            ),
            analyses.panel_correlations(
                dataset, log_values, config_module.MANUSCRIPT_INTERACTOME,
                "GeneMANIA interactome",
            ),
            analyses.panel_correlations(
                dataset, log_values, config_module.CANCER_TESTIS_ANTIGENS,
                "cancer-testis antigens",
            ),
        ],
        ignore_index=True,
    )

    # ------------------------------------------- controllo di confondimento #
    stratified_features: dict[str, pd.Series] = {}
    for gene in ["TEKT1", "CFAP52", "HYDIN", "SPEF2", "SPA17", "FOXJ1"]:
        symbol = data_loading.resolve_symbols(dataset.log_expression.index, [gene])
        if symbol:
            stratified_features[symbol] = dataset.log_expression.loc[symbol]

    # -------------------------------------------------------------- differenziale #
    differential = analyses.differential_expression(
        dataset, expressed, detection_group, "positive", "negative"
    )
    differential_median = analyses.differential_expression(
        dataset, expressed, median_group, "high", "low"
    )

    # ---------------------------------------------------------------- immuno #
    scores, signature_composition = analyses.signature_scores(
        dataset, config_module.IMMUNE_SIGNATURES
    )
    checkpoints, signature_stats = analyses.immune_associations(
        dataset, log_values, config_module.IMMUNE_CHECKPOINTS, scores, detection_group
    )

    stratified_features["Ciliated cell program (score)"] = scores["Ciliated cell program"]
    stratified_features["CD4+ T cells (score)"] = scores["CD4+ T cells"]
    stratified_features["Macrophages (score)"] = scores["Macrophages"]
    stratified = analyses.stratified_correlations(
        dataset, log_values, stratified_features, config_module.HISTOTYPE_ORDER
    )

    # ----------------------------------------------------------- arricchimento #
    enrichment_tables: dict[str, pd.DataFrame] = {}
    enrichment_summary: dict[str, Any] = {}
    if cfg.offline:
        LOGGER.warning("Arricchimento GO saltato (--skip-enrichment)")
        enrichment_summary["stato"] = "saltato (--skip-enrichment)"
    else:
        try:
            gene2go = enrichment.download_human_gene2go(
                config_module.GENE2GO_URL,
                cfg.cache_dir / "gene2go_human.tsv.gz",
                config_module.HUMAN_TAX_ID,
            )
            gene_info = cfg.cache_dir / "Homo_sapiens.gene_info.gz"
            if not gene_info.is_file():
                raise enrichment.EnrichmentError(
                    f"File {gene_info.name} assente: eseguire prima la pipeline "
                    f"GSE101108 (scarica l'annotazione NCBI in data/cache)."
                )
            ontology = enrichment.load_gene_ontology(gene2go, gene_info, expressed)

            coexpressed_genes = coexpression.loc[
                (coexpression["q_value_BH"] < cfg.fdr_alpha)
                & (coexpression["rho"] >= cfg.coexpression_rho)
            ].head(cfg.coexpression_top_n).index.tolist()
            enrichment_tables["GO_coexpressed"] = enrichment.overrepresentation(
                coexpressed_genes, ontology,
                config_module.GO_MIN_GENES, config_module.GO_MAX_GENES,
            )
            enrichment_summary["n_geni_co-espressi_testati"] = len(coexpressed_genes)

            if not differential.empty:
                up_genes = differential.loc[
                    (differential["q_value_BH"] < cfg.fdr_alpha)
                    & (differential["log2FC"] >= cfg.de_log2fc)
                ].index.tolist()
                enrichment_tables["GO_up_in_positive"] = enrichment.overrepresentation(
                    up_genes, ontology,
                    config_module.GO_MIN_GENES, config_module.GO_MAX_GENES,
                )
                enrichment_summary["n_geni_up_testati"] = len(up_genes)

            for name, table in enrichment_tables.items():
                enrichment_summary[f"{name}_termini_significativi"] = (
                    int(table["significant"].sum()) if not table.empty else 0
                )
                if not table.empty and table["significant"].any():
                    best = table[table["significant"]].sort_values("p_value").iloc[0]
                    enrichment_summary[f"{name}_top_term"] = (
                        f"{best['go_term']} ({best['category']}, q={best['q_value_BH']:.2e})"
                    )
        except enrichment.EnrichmentError as exc:
            LOGGER.error("%s", exc)
            enrichment_summary["stato"] = f"non disponibile: {exc}"

    # ---------------------------------------------------------------- figure #
    figures.plot_target_by_histotype(
        profile.per_sample, profile.detection_summary, profile.expression_pairwise,
        profile.detection_pairwise, profile.kruskal, config_module.HISTOTYPE_ORDER,
        cfg.target_gene, cfg.figures_dir / f"Figure_{cfg.target_gene}_1_histotype.png",
        cfg.figure_dpi,
    )
    figures.plot_clinical_associations(
        profile.per_sample, profile.stage, profile.age, cfg.target_gene,
        cfg.figures_dir / f"Figure_{cfg.target_gene}_2_clinical.png", cfg.figure_dpi,
    )
    figures.plot_coexpression(
        coexpression, panels[panels["panel"] == "GEPIA2 top-10 (TCGA)"], cfg.target_gene,
        cfg.figures_dir / f"Figure_{cfg.target_gene}_3_coexpression.png", cfg.figure_dpi,
    )
    figures.plot_volcano(
        differential, cfg.target_gene,
        cfg.figures_dir / f"Figure_{cfg.target_gene}_4_volcano.png",
        cfg.figure_dpi, cfg.de_log2fc,
    )
    if "GO_coexpressed" in enrichment_tables:
        figures.plot_go_enrichment(
            enrichment.top_terms(enrichment_tables["GO_coexpressed"], config_module.GO_TOP_TERMS),
            cfg.figures_dir / f"Figure_{cfg.target_gene}_5_GO.png", cfg.figure_dpi,
        )
    figures.plot_immune(
        checkpoints, signature_stats, cfg.target_gene,
        cfg.figures_dir / f"Figure_{cfg.target_gene}_6_immune.png", cfg.figure_dpi,
    )
    figures.plot_signature_heatmap(
        scores, profile.per_sample, config_module.HISTOTYPE_ORDER,
        cfg.figures_dir / f"Figure_{cfg.target_gene}_7_signature_heatmap.png", cfg.figure_dpi,
    )

    # --------------------------------------------------------------- output #
    results = _collect_results(
        cfg, dataset, expressed, profile, coexpression, panels, differential,
        differential_median, checkpoints, signature_stats, enrichment_summary,
        stratified,
    )
    numbers = reporting.build_manuscript_numbers(results)

    reporting.save_workbook(
        {
            "detection": profile.detection_summary,
            "detection_sensibilita": profile.detection_sensitivity,
            "detection_confronti": profile.detection_pairwise,
            "espressione": profile.expression_summary,
            "espressione_confronti": profile.expression_pairwise,
            "stadio": profile.stage,
            "eta_gruppi": profile.age_groups,
            "per_campione": profile.per_sample,
            "pannelli_pubblicati": panels,
            "coespressione_top": coexpression.head(500).reset_index(),
            "differenziale_top": differential.head(500).reset_index(),
            "differenziale_mediana": differential_median.head(500).reset_index(),
            "checkpoint_immunitari": checkpoints,
            "firme_immunitarie": signature_stats,
            "firme_composizione": signature_composition,
            "stratificato_per_istotipo": stratified,
            **{k: enrichment.top_terms(v, 50) for k, v in enrichment_tables.items()},
        },
        cfg.tables_dir / f"{cfg.target_gene}_{cfg.gse_id}_results.xlsx",
    )
    reporting.save_csv(coexpression, cfg.tables_dir / f"{cfg.target_gene}_coexpression_all.csv.gz")
    if not differential.empty:
        reporting.save_csv(
            differential, cfg.tables_dir / f"{cfg.target_gene}_differential_all.csv.gz"
        )
    for name, table in enrichment_tables.items():
        if not table.empty:
            reporting.save_csv(table, cfg.tables_dir / f"{name}.csv", index=False)
    reporting.save_csv(scores, cfg.tables_dir / "immune_signature_scores.csv")

    reporting.save_json(
        {"numbers": numbers, "config": cfg.as_dict(),
         "durata_secondi": round(time.time() - started, 1)},
        cfg.tables_dir / "manuscript_numbers.json",
    )
    reporting.save_text(
        reporting.summary_lines(numbers), cfg.tables_dir / "analysis_summary.txt"
    )

    LOGGER.info(
        "Analisi completata in %.1f s. Tabelle in %s, figure in %s",
        time.time() - started, cfg.tables_dir, cfg.figures_dir,
    )
    _log_headline(profile, cfg.target_gene)
    return 0


def _collect_results(
    cfg: AnalysisConfig,
    dataset: data_loading.Dataset,
    expressed: pd.Index,
    profile: analyses.TargetProfile,
    coexpression: pd.DataFrame,
    panels: pd.DataFrame,
    differential: pd.DataFrame,
    differential_median: pd.DataFrame,
    checkpoints: pd.DataFrame,
    signature_stats: pd.DataFrame,
    enrichment_summary: dict[str, Any],
    stratified: pd.DataFrame,
) -> dict[str, Any]:
    """Collect the results into a dictionary for the report."""
    library = dataset.metadata["library_size"]
    significant_signatures = signature_stats[signature_stats["significant"].fillna(False)]
    return {
        "gse_id": cfg.gse_id,
        "n_samples": dataset.n_samples,
        "n_genes_total": dataset.n_genes,
        "n_genes_expressed": len(expressed),
        "histotype_counts": dataset.metadata["histotype"].value_counts().sort_index().to_dict(),
        "median_library_size": float(library.median()),
        "library_size_range": [float(library.min()), float(library.max())],
        "profile": profile,
        "coexpression_summary": {
            "n_tested": int(len(coexpression)),
            "n_significant": int((coexpression["q_value_BH"] < cfg.fdr_alpha).sum()),
            "n_above_rho": int(
                ((coexpression["q_value_BH"] < cfg.fdr_alpha)
                 & (coexpression["rho"] >= cfg.coexpression_rho)).sum()
            ),
            "top_genes": coexpression.head(20).reset_index().to_dict("records"),
        },
        "panel_summary": {
            panel: {
                "n_found": int(group["found"].sum()),
                "n_total": int(len(group)),
                "n_significant": int(group["significant"].fillna(False).sum()),
                "median_rho": float(group.loc[group["found"], "rho"].median())
                if group["found"].any() else None,
                "genes": group.loc[group["found"], ["published_symbol", "rho", "q_value_BH"]]
                .to_dict("records"),
            }
            for panel, group in panels.groupby("panel")
        },
        "differential_summary": {
            "comparison": f"{cfg.target_gene}-positive vs negative (>= {cfg.detection_threshold} counts)",
            "n_tested": int(len(differential)),
            "n_significant": int((differential["q_value_BH"] < cfg.fdr_alpha).sum())
            if not differential.empty else 0,
            "n_up": int(((differential["q_value_BH"] < cfg.fdr_alpha)
                         & (differential["log2FC"] >= cfg.de_log2fc)).sum())
            if not differential.empty else 0,
            "n_down": int(((differential["q_value_BH"] < cfg.fdr_alpha)
                           & (differential["log2FC"] <= -cfg.de_log2fc)).sum())
            if not differential.empty else 0,
            "top_genes": differential.head(20).reset_index().to_dict("records")
            if not differential.empty else [],
            "sensitivity_median_split_n_significant": int(
                (differential_median["q_value_BH"] < cfg.fdr_alpha).sum()
            ) if not differential_median.empty else 0,
        },
        "enrichment_summary": enrichment_summary,
        "stratified_summary": stratified.to_dict("records") if not stratified.empty else [],
        "immune_summary": {
            "checkpoint_significativi": checkpoints.loc[
                checkpoints["significant"].fillna(False), "published_symbol"
            ].tolist(),
            "checkpoint_rho": {
                row["published_symbol"]: round(float(row["rho"]), 3)
                for _, row in checkpoints[checkpoints["found"]].iterrows()
            },
            "firme_significative": significant_signatures["signature"].tolist(),
            "firme_rho": {
                row["signature"]: round(float(row["rho"]), 3)
                for _, row in signature_stats.iterrows()
            },
        },
    }


def _log_headline(profile: analyses.TargetProfile, target: str) -> None:
    """Stampa a console i risultati principali."""
    LOGGER.info("-" * 70)
    LOGGER.info("RISULTATI PRINCIPALI")
    for _, row in profile.detection_summary.iterrows():
        LOGGER.info(
            "  %-22s %s-positivi: %2d/%-3d (%5.1f%%)",
            row["histotype"], target, row["n_positive"], row["n"], row["detection_percent"],
        )
    LOGGER.info("  Kruskal-Wallis (livelli): p = %s", format_p(profile.kruskal["p_value"]))
    significant = profile.detection_pairwise[profile.detection_pairwise["q_value_BH"] < 0.05]
    for _, row in significant.iterrows():
        LOGGER.info(
            "  Rilevabilita' %s vs %s: %.0f%% vs %.0f%%, q = %s",
            row["group_1"], row["group_2"], 100 * row["rate_1"], 100 * row["rate_2"],
            format_p(row["q_value_BH"]),
        )
    LOGGER.info("-" * 70)


def main(argv: list[str] | None = None) -> int:
    """Entry point with error handling."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
        return 130
    except data_loading.DataError as exc:
        logging.getLogger("spef1").exception("Dati non disponibili")
        print(f"\nERRORE SUI DATI: {exc}", file=sys.stderr)
        return 2
    except reporting.ReportError as exc:
        logging.getLogger("spef1").exception("Errore di scrittura")
        print(f"\nERRORE DI SCRITTURA: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # errori imprevisti
        logging.getLogger("spef1").exception("Errore non gestito")
        print(f"\nERRORE INATTESO: {exc}\nDettagli in logs/spef1_analysis.log",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
