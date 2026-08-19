"""Configuration of the SPEF1 bioinformatic analysis on GSE101108.

The analysis consumes the outputs of the ``GSE101108_pipeline`` and produces
tables and figures ready for the manuscript.

"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------- #
# Parametri principali                                                         #
# --------------------------------------------------------------------------- #

#: Target gene of the study.
TARGET_GENE = "SPEF1"

#: Serie GEO di riferimento.
GSE_ID = "GSE101108"

#: Histotypes considered in the analyses (order used in the figures).
HISTOTYPE_ORDER = ["Serous", "Clear cell", "Endometrioid", "Mucinous"]

#: Detection threshold (raw counts) that defines a sample as
#: "SPEF1-positive". Chosen a priori: 5 reads is the minimum for which the
#: segnale/rumore di htseq-count su librerie da ~6 M read e' accettabile.
DETECTION_THRESHOLD = 5

#: Soglie usate nelle analisi di sensibilita'.
DETECTION_THRESHOLDS_SENSITIVITY = [1, 3, 5, 10]

#: Expression filter for the transcriptomic analyses: a gene is analyzed
#: if it has at least ``MIN_COUNTS`` counts in at least ``MIN_FRACTION`` of the samples.
MIN_COUNTS = 10
MIN_FRACTION = 0.25

#: Soglie di significativita'.
FDR_ALPHA = 0.05

#: Co-expression: genes considered "correlated" with SPEF1.
COEXPRESSION_RHO = 0.40
COEXPRESSION_TOP_N = 300

#: Espressione differenziale SPEF1-positivi vs negativi.
DE_LOG2FC = 1.0

#: GO enrichment: allowed term sizes.
GO_MIN_GENES = 5
GO_MAX_GENES = 1000
GO_TOP_TERMS = 15

#: Source of the GO annotations (the only host reachable on many hospital/
#: corporate networks; the file is filtered in streaming mode on tax_id 9606).
GENE2GO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)
HUMAN_TAX_ID = "9606"

# --------------------------------------------------------------------------- #
# Reference gene panels
# --------------------------------------------------------------------------- #

#: Top 10 genes correlated with SPEF1 reported in the manuscript (GEPIA2 on TCGA).
#: In parentheses the current HGNC symbols used to search the matrix.
MANUSCRIPT_GEPIA_GENES = {
    "C5orf49": ["C5ORF49", "SMIM43"],
    "CFAP52": ["CFAP52", "WDR16"],
    "TEKT1": ["TEKT1"],
    "C9orf171": ["C9ORF171", "CFAP77"],
    "C6orf118": ["C6ORF118", "CFAP206"],
    "C9orf116": ["C9ORF116", "PIERCE1"],
    "SPAG8": ["SPAG8"],
    "TTLL9": ["TTLL9"],
    "LRRC43": ["LRRC43"],
    "RSPH1": ["RSPH1"],
}

#: SPEF1 interactors reported in the manuscript (GeneMANIA/STRING).
MANUSCRIPT_INTERACTOME = [
    "MAPRE1", "MAPRE3", "CFAP70", "SPAG16", "TSSK4", "SPEF2",
    "EFHC1", "SPAG6", "HYDIN", "ODF2", "SPATA4",
]

#: Other cancer-testis antigens mentioned in the manuscript.
CANCER_TESTIS_ANTIGENS = ["SPA17", "SPAG17", "MAGEA1", "MAGEA4", "CTAG1B", "PRAME", "TEX19"]

#: Immune checkpoints analyzed in the manuscript (TIMER/ssGSEA).
IMMUNE_CHECKPOINTS = ["CD274", "CTLA4", "LAG3", "PDCD1", "TIGIT", "HAVCR2", "PDCD1LG2"]

#: Marker signatures for the estimation of immune infiltration.
#: This is NOT a deconvolution: it is a marker-based score (mean of the
#: marker z-scores), declared as such in the Methods.
IMMUNE_SIGNATURES: dict[str, list[str]] = {
    "B cells": ["CD19", "MS4A1", "CD79A", "CD79B", "BLNK"],
    "CD8+ T cells": ["CD8A", "CD8B", "GZMK", "CD3D", "CD3E"],
    "CD4+ T cells": ["CD4", "IL7R", "CD40LG", "CD3D", "CD3G"],
    "Macrophages": ["CD68", "CD163", "MSR1", "MRC1", "CSF1R"],
    "Neutrophils": ["FCGR3B", "CSF3R", "S100A8", "S100A9", "CXCR2"],
    "Dendritic cells": ["ITGAX", "CD1C", "CLEC9A", "LAMP3", "FLT3"],
    "NK cells": ["NCR1", "KLRD1", "NKG7", "GNLY", "PRF1"],
    "Cytolytic activity": ["GZMA", "PRF1", "GZMB", "GNLY"],
    "Ciliated cell program": ["FOXJ1", "TPPP3", "PIFO", "DNAI1", "SPAG6"],
}

#: Epithelial/histotype markers used as a biological quality control.
HISTOTYPE_MARKERS = {
    "Clear cell": ["HNF1B", "NAPSA", "VIM"],
    "Endometrioid": ["ESR1", "PGR", "VIM"],
    "Mucinous": ["MUC2", "CDX2", "TFF1", "TFF3"],
    "Serous": ["WT1", "PAX8", "TP53", "MUC16"],
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisConfig:
    """Parametri e percorsi dell'analisi."""

    target_gene: str = TARGET_GENE
    gse_id: str = GSE_ID
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    pipeline_root: Path | None = None
    detection_threshold: int = DETECTION_THRESHOLD
    min_counts: int = MIN_COUNTS
    min_fraction: float = MIN_FRACTION
    fdr_alpha: float = FDR_ALPHA
    coexpression_rho: float = COEXPRESSION_RHO
    coexpression_top_n: int = COEXPRESSION_TOP_N
    de_log2fc: float = DE_LOG2FC
    figure_dpi: int = 300
    include_unspecified: bool = False
    offline: bool = False

    def __post_init__(self) -> None:
        if self.pipeline_root is None:
            self.pipeline_root = self.project_root.parent / "GSE101108_pipeline"
        self.pipeline_root = Path(self.pipeline_root)

    # ----------------------------------------------------------- input paths #
    @property
    def counts_file(self) -> Path:
        return (
            self.pipeline_root
            / "data/processed"
            / f"{self.gse_id}_complete_expression_matrix.csv.gz"
        )

    @property
    def metadata_file(self) -> Path:
        return (
            self.pipeline_root / "data/metadata" / f"{self.gse_id}_metadata_normalized.xlsx"
        )

    @property
    def histotype_file(self) -> Path:
        return (
            self.pipeline_root
            / "data/metadata"
            / f"{self.gse_id}_histotype_classification.xlsx"
        )

    @property
    def cache_dir(self) -> Path:
        return self.pipeline_root / "data/cache"

    # ---------------------------------------------------------- output paths #
    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def manuscript_dir(self) -> Path:
        return self.project_root / "manuscript"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "spef1_analysis.log"

    def create_directories(self) -> None:
        """Create the output directories."""
        for directory in (
            self.tables_dir,
            self.figures_dir,
            self.manuscript_dir,
            self.logs_dir,
            self.cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return {
            "target_gene": self.target_gene,
            "gse_id": self.gse_id,
            "detection_threshold": self.detection_threshold,
            "min_counts": self.min_counts,
            "min_fraction": self.min_fraction,
            "fdr_alpha": self.fdr_alpha,
            "coexpression_rho": self.coexpression_rho,
            "coexpression_top_n": self.coexpression_top_n,
            "de_log2fc": self.de_log2fc,
            "include_unspecified": self.include_unspecified,
            "offline": self.offline,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnalysisConfig":
        """Build the configuration from a dictionary (YAML/CLI)."""
        valid = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in payload.items():
            key_norm = str(key).strip().lower().replace("-", "_")
            if key_norm in valid and value is not None:
                if key_norm in {"project_root", "pipeline_root"}:
                    value = Path(value)
                kwargs[key_norm] = value
        return cls(**kwargs)
