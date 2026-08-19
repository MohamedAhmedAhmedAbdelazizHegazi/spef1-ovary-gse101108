"""Gene Ontology functional enrichment with local annotations.

The annotations come from NCBI's ``gene2go``: the full file (~1.3 GB, all
species) is decompressed in streaming mode and filtered on ``tax_id 9606``,
caching only the ~450,000 human rows (~5 MB). This avoids depending on web
enrichment services, often unreachable from hospital networks, and makes the
analysis fully reproducible.

The over-representation test is hypergeometric (equivalent to the one-tailed
Fisher exact test), with Benjamini-Hochberg FDR correction.

"""

from __future__ import annotations

import gzip
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
from scipy import stats

from .statistics import benjamini_hochberg

LOGGER = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "Process": "Biological process",
    "Function": "Molecular function",
    "Component": "Cellular component",
}


class EnrichmentError(RuntimeError):
    """GO annotations not available."""


@dataclass
class GeneOntology:
    """Human GO annotations ready for the enrichment analysis."""

    annotations: pd.DataFrame  # colonne: symbol, go_id, go_term, category
    background: set[str]

    def sets_by_category(self, category: str) -> dict[str, tuple[str, set[str]]]:
        """Map ``go_id -> (term name, set of symbols)``."""
        subset = self.annotations[self.annotations["category"] == category]
        grouped = subset.groupby(["go_id", "go_term"])["symbol"].apply(set)
        return {go_id: (term, genes) for (go_id, term), genes in grouped.items()}


# --------------------------------------------------------------------------- #
# Acquisition of the annotations
# --------------------------------------------------------------------------- #


def download_human_gene2go(
    url: str, cache_file: Path, tax_id: str = "9606", timeout: int = 300
) -> Path:
    """Download ``gene2go`` filtering the human-species rows in streaming mode.

    The file is sorted by increasing ``tax_id``: the download is stopped as soon
    as the requested tax_id is passed, avoiding downloading the whole archive.

    """
    cache_file = Path(cache_file)
    if cache_file.is_file() and cache_file.stat().st_size > 1_000_000:
        LOGGER.info("Annotazioni GO gia' in cache: %s", cache_file.name)
        return cache_file

    LOGGER.info("Download delle annotazioni GO da NCBI (filtro tax_id=%s)...", tax_id)
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    tail = ""
    kept = 0
    downloaded = 0
    stop = False
    temp_file = cache_file.with_suffix(cache_file.suffix + ".part")

    try:
        with requests.get(
            url, stream=True, timeout=timeout, headers={"User-Agent": "SPEF1-analysis/1.0"}
        ) as response:
            response.raise_for_status()
            with gzip.open(temp_file, "wt", encoding="utf-8") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    downloaded += len(chunk)
                    text = tail + decompressor.decompress(chunk).decode("utf-8", "replace")
                    lines = text.split("\n")
                    tail = lines.pop()
                    for line in lines:
                        if line.startswith("#"):
                            handle.write(line + "\n")
                            continue
                        current = line.split("\t", 1)[0]
                        if current == tax_id:
                            handle.write(line + "\n")
                            kept += 1
                        elif kept and current.isdigit() and int(current) > int(tax_id):
                            stop = True
                            break
                    if stop:
                        break
    except (requests.RequestException, zlib.error, OSError) as exc:
        temp_file.unlink(missing_ok=True)
        raise EnrichmentError(
            f"Impossibile scaricare le annotazioni GO ({exc}). Senza connessione "
            f"l'analisi di arricchimento viene saltata: usare --skip-enrichment "
            f"oppure copiare manualmente {cache_file.name} nella cartella cache."
        ) from exc

    temp_file.replace(cache_file)
    LOGGER.info(
        "Annotazioni GO umane salvate: %d righe (%.0f MB scaricati, %.1f MB in cache)",
        kept,
        downloaded / 1e6,
        cache_file.stat().st_size / 1e6,
    )
    return cache_file


def load_gene_ontology(
    gene2go_file: Path, gene_info_file: Path, background: Iterable[str]
) -> GeneOntology:
    """Load the GO annotations and map them onto the gene symbols.

    Args:
        gene2go_file: filtered human file produced by :func:`download_human_gene2go`.
        gene_info_file: ``Homo_sapiens.gene_info.gz`` (for Entrez -> symbol).
        background: gene symbols tested in the analysis (test universe).

    """
    gene2go = pd.read_csv(
        gene2go_file,
        sep="\t",
        compression="gzip",
        comment="#",
        header=None,
        names=[
            "tax_id", "GeneID", "GO_ID", "Evidence", "Qualifier", "GO_term",
            "PubMed", "Category",
        ],
        dtype=str,
    )
    # negative annotations ("NOT ...") do not indicate membership of the term
    gene2go = gene2go[~gene2go["Qualifier"].fillna("").str.upper().str.startswith("NOT")]

    gene_info = pd.read_csv(
        gene_info_file,
        sep="\t",
        compression="gzip",
        usecols=["GeneID", "Symbol"],
        dtype=str,
    )
    symbol_by_entrez = dict(zip(gene_info["GeneID"], gene_info["Symbol"].str.upper()))
    gene2go["symbol"] = gene2go["GeneID"].map(symbol_by_entrez)
    gene2go = gene2go.dropna(subset=["symbol"])

    universe = {str(g).upper() for g in background}
    annotations = gene2go.loc[
        gene2go["symbol"].isin(universe),
        ["symbol", "GO_ID", "GO_term", "Category"],
    ].rename(columns={"GO_ID": "go_id", "GO_term": "go_term", "Category": "category"})
    annotations = annotations.drop_duplicates()

    LOGGER.info(
        "Annotazioni GO caricate: %d coppie gene-termine su %d geni "
        "dell'universo di test",
        len(annotations),
        annotations["symbol"].nunique(),
    )
    return GeneOntology(annotations=annotations, background=universe)


# --------------------------------------------------------------------------- #
# Over-representation test
# --------------------------------------------------------------------------- #


def overrepresentation(
    gene_set: Sequence[str],
    ontology: GeneOntology,
    min_genes: int = 5,
    max_genes: int = 1000,
) -> pd.DataFrame:
    """Hypergeometric GO over-representation test.

    Args:
        gene_set: genes of interest (must belong to the test universe).
        ontology: annotations loaded with :func:`load_gene_ontology`.
        min_genes / max_genes: allowed size of the GO terms.

    Returns:
        Table with p-value, BH q-value, fold enrichment and overlapping genes.

    """
    query = {str(g).upper() for g in gene_set} & ontology.background
    if not query:
        LOGGER.warning("Nessun gene della lista appartiene all'universo di test")
        return pd.DataFrame()

    universe_size = len(ontology.background)
    rows = []
    for category, label in CATEGORY_LABELS.items():
        for go_id, (term, genes) in ontology.sets_by_category(category).items():
            genes = genes & ontology.background
            if not (min_genes <= len(genes) <= max_genes):
                continue
            overlap = genes & query
            if len(overlap) < 2:
                continue
            pvalue = stats.hypergeom.sf(
                len(overlap) - 1, universe_size, len(genes), len(query)
            )
            expected = len(genes) * len(query) / universe_size
            rows.append(
                {
                    "category": label,
                    "go_id": go_id,
                    "go_term": term,
                    "n_genes_in_term": len(genes),
                    "n_overlap": len(overlap),
                    "expected": round(expected, 2),
                    "fold_enrichment": round(len(overlap) / expected, 2) if expected else np.nan,
                    "p_value": pvalue,
                    "genes": ", ".join(sorted(overlap)[:40]),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        LOGGER.warning("Nessun termine GO testabile per questa lista di geni")
        return table

    # correction per family of tests (one family per GO category)
    table["q_value_BH"] = np.nan
    for category in table["category"].unique():
        mask = table["category"] == category
        table.loc[mask, "q_value_BH"] = benjamini_hochberg(
            table.loc[mask, "p_value"].to_numpy()
        )
    table["significant"] = table["q_value_BH"] < 0.05
    table = table.sort_values(["category", "p_value"])
    LOGGER.info(
        "Arricchimento GO: %d termini testati, %d significativi (q<0.05) su %d geni",
        len(table),
        int(table["significant"].sum()),
        len(query),
    )
    return table


def top_terms(table: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """First ``n`` terms per category, sorted by p-value."""
    if table.empty:
        return table
    return (
        table.sort_values("p_value")
        .groupby("category", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )
