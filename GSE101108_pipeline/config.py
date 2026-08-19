"""Central configuration of the GSE101108 pipeline.

Every user-editable parameter is collected here. The same configuration can be
overridden by an external YAML file (``--config config.yaml``) or by
command-line arguments.

"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Valori di default                                                            #
# --------------------------------------------------------------------------- #

GSE_ID = "GSE101108"

#: Genes extracted from the expression matrix. Edit this list freely
#: (or use ``--genes A B C`` / the ``genes_of_interest`` key in the YAML).
GENES_OF_INTEREST = [
    "SPEF1",  # primary gene of interest for this project
    "ARID1A",
    "PIK3CA",
    "PTEN",
    "KRAS",
    "HNF1B",
    "TP53",
    "BRCA1",
    "BRCA2",
    "CCNE1",
    "PARP1",
    "VEGFA",
]

#: Histotypes kept in the main final dataset.
#: Serous carcinomas are included so that the full case series is available to
#: the downstream analyses (between-histotype comparisons, serous as a reference
#: group). The non-serous subset is still exported separately,
#: in ``*_non_serous_expression_matrix.csv.gz``.
ALLOWED_HISTOTYPES = [
    "Clear cell",
    "Endometrioid",
    "Mucinous",
    "Serous",
]

#: Histotypes always excluded from the final dataset.
#: To go back to the non-serous-only dataset, add "Serous" here
#: (oppure usare: python main.py --histotypes "Clear cell" Endometrioid Mucinous).
EXCLUDED_HISTOTYPES: list[str] = []

#: Label assigned to histotype values that are not recognized.
UNSPECIFIED_HISTOTYPE = "Other or unspecified"

#: URL of the GEO supplementary-file directory on NCBI.
GEO_SUPPL_URL_TEMPLATE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_dir}/{gse_id}/suppl/"
)
GEO_SOFT_URL_TEMPLATE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_dir}/{gse_id}/soft/"
    "{gse_id}_family.soft.gz"
)
GEO_MATRIX_URL_TEMPLATE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_dir}/{gse_id}/matrix/"
    "{gse_id}_series_matrix.txt.gz"
)

#: Terms used to locate the histotype column/field in the metadata.
HISTOTYPE_FIELD_KEYWORDS = [
    "histotype",
    "histology",
    "histological type",
    "histological subtype",
    "tumor type",
    "tumour type",
    "subtype",
    "diagnosis",
]

#: Candidate column names for the gene identifier in the matrix.
GENE_ID_COLUMN_CANDIDATES = [
    "gene_symbol",
    "symbol",
    "gene_name",
    "gene",
    "genes",
    "ensembl_gene_id",
    "ensembl_id",
    "ensembl",
    "gene_id",
    "geneid",
    "id",
    "feature",
    "feature_id",
    "name",
]

#: Selection priority of the expression matrix (higher score
#: = preferito). Vedi ``src/expression_loader.py``.
MATRIX_PRIORITY_KEYWORDS = {
    "raw_counts": 100,
    "counts": 90,
    "tpm": 70,
    "fpkm": 60,
    "rpkm": 55,
    "cpm": 50,
    "expression": 40,
    "normalized": 35,
}

#: Accepted extensions for tabular supplementary files.
TABULAR_EXTENSIONS = {".txt", ".tsv", ".csv", ".tab", ".gz", ".xlsx", ".xls"}

#: Minimum share of allowed samples that must be correctly matched to the matrix.
MIN_MATCH_FRACTION = 0.80

#: Figure resolution.
FIGURE_DPI = 300

#: Numero massimo di righe scritte in un singolo foglio Excel.
EXCEL_MAX_ROWS = 1_000_000


# --------------------------------------------------------------------------- #
# Configuration object
# --------------------------------------------------------------------------- #


def _series_dir(gse_id: str) -> str:
    """Return the GEO FTP sub-directory (e.g. ``GSE101nnn``)."""
    digits = "".join(ch for ch in gse_id if ch.isdigit())
    if len(digits) <= 3:
        return f"GSE{digits}nnn"
    return f"GSE{digits[:-3]}nnn"


@dataclass
class PipelineConfig:
    """Container for all pipeline parameters."""

    gse_id: str = GSE_ID
    genes_of_interest: list[str] = field(
        default_factory=lambda: list(GENES_OF_INTEREST)
    )
    allowed_histotypes: list[str] = field(
        default_factory=lambda: list(ALLOWED_HISTOTYPES)
    )
    excluded_histotypes: list[str] = field(
        default_factory=lambda: list(EXCLUDED_HISTOTYPES)
    )
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    keep_unspecified: bool = False
    use_mygene: bool = True
    use_ncbi_fallback: bool = True
    min_match_fraction: float = MIN_MATCH_FRACTION
    force: bool = False
    download_timeout: int = 120
    download_retries: int = 3
    figure_dpi: int = FIGURE_DPI

    # ----------------------------------------------------------------- paths #
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def manual_gene_map(self) -> Path:
        """Mappa manuale ``identificativo -> simbolo`` (facoltativa)."""
        return self.metadata_dir / "manual_gene_map.csv"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.gse_id}_pipeline.log"

    # ------------------------------------------------------------------ urls #
    @property
    def suppl_url(self) -> str:
        return GEO_SUPPL_URL_TEMPLATE.format(
            series_dir=_series_dir(self.gse_id), gse_id=self.gse_id
        )

    @property
    def soft_url(self) -> str:
        return GEO_SOFT_URL_TEMPLATE.format(
            series_dir=_series_dir(self.gse_id), gse_id=self.gse_id
        )

    @property
    def matrix_url(self) -> str:
        return GEO_MATRIX_URL_TEMPLATE.format(
            series_dir=_series_dir(self.gse_id), gse_id=self.gse_id
        )

    # --------------------------------------------------------------- helpers #
    def create_directories(self) -> None:
        """Create all working directories if they do not exist."""
        for directory in (
            self.raw_dir,
            self.metadata_dir,
            self.processed_dir,
            self.reports_dir,
            self.figures_dir,
            self.logs_dir,
            self.cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def out(self, subdir: Path, suffix: str) -> Path:
        """Build an output path prefixed with the GSE ID."""
        return subdir / f"{self.gse_id}_{suffix}"

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load the configuration from an external YAML file.

        Unknown keys are ignored (the caller is expected to warn about them);
        missing keys keep their default values.

        """
        import yaml  # import locale: dipendenza opzionale a runtime

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File di configurazione non trovato: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload: Mapping[str, Any] = yaml.safe_load(handle) or {}
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PipelineConfig":
        """Build the configuration from a dictionary."""
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            key_norm = str(key).strip().lower().replace("-", "_")
            if key_norm not in valid:
                continue
            if key_norm == "project_root":
                value = Path(value)
            kwargs[key_norm] = value
        return cls(**kwargs)

    def update(self, **kwargs: Any) -> "PipelineConfig":
        """Update the non-null fields in place and return ``self``."""
        valid = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if value is None or key not in valid:
                continue
            setattr(self, key, value)
        return self

    def as_dict(self) -> dict[str, Any]:
        """Serializable representation of the configuration."""
        return {
            "gse_id": self.gse_id,
            "genes_of_interest": list(self.genes_of_interest),
            "allowed_histotypes": list(self.allowed_histotypes),
            "excluded_histotypes": list(self.excluded_histotypes),
            "keep_unspecified": self.keep_unspecified,
            "use_mygene": self.use_mygene,
            "use_ncbi_fallback": self.use_ncbi_fallback,
            "min_match_fraction": self.min_match_fraction,
            "force": self.force,
            "project_root": str(self.project_root),
        }


def normalize_gene_list(genes: Sequence[str]) -> list[str]:
    """Clean up and upper-case a list of gene symbols."""
    seen: set[str] = set()
    result: list[str] = []
    for gene in genes:
        symbol = str(gene).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result
