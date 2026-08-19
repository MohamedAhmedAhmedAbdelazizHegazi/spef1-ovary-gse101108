"""GSE101108 pipeline package.

Modules:
    downloader          - download of the GEO files (supplementary, SOFT, matrix)
    geo_metadata        - parsing of the GEO/GSM metadata
    filtering           - histotype normalization and sample selection
    expression_loader   - detection and reading of the expression matrix
    gene_annotation     - recognition/conversion of gene identifiers
    sample_matching     - matching of matrix columns <-> GEO samples
    quality_control     - quality controls and figures
    exporters           - writing of CSV/Excel/JSON

"""

from __future__ import annotations

__all__ = [
    "downloader",
    "geo_metadata",
    "filtering",
    "expression_loader",
    "gene_annotation",
    "sample_matching",
    "quality_control",
    "exporters",
]

__version__ = "1.0.0"
