"""SPEF1 bioinformatic analysis package on GSE101108.

Modules:
    data_loading  - loading, alignment and normalization of the counts
    statistics    - non-parametric tests, FDR, effect size
    analyses      - target profile, co-expression, differential, immune
    enrichment    - local GO annotations and over-representation testing
    figures       - figures for the manuscript
    reporting     - tables, citable numbers and summaries

"""

from __future__ import annotations

__all__ = [
    "data_loading",
    "statistics",
    "analyses",
    "enrichment",
    "figures",
    "reporting",
]

__version__ = "1.0.0"
