"""Recognition, cleaning and conversion of gene identifiers.

Identifiers may be HGNC symbols, Ensembl Gene IDs (with or without version),
Entrez Gene IDs or other. Conversion to gene symbols uses ``mygene``; the
results are cached on disk to make later runs reproducible even offline.

"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Fallback annotation, used when mygene.info is unreachable
#: (corporate proxies that block the service's TLS certificate). The file is
#: hosted on the same host as the GEO data.
NCBI_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)

ENSEMBL_GENE_RE = re.compile(r"^(ENS[A-Z]*G\d{6,})(\.\d+)?$", re.IGNORECASE)
ENTREZ_RE = re.compile(r"^\d+$")
SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-\.@_]{0,30}$")

ID_TYPE_ENSEMBL = "ensembl_gene_id"
ID_TYPE_ENTREZ = "entrez_gene_id"
ID_TYPE_SYMBOL = "gene_symbol"
ID_TYPE_OTHER = "other"


@dataclass
class AggregationResult:
    """Outcome of the aggregation of duplicate rows of the same gene."""

    matrix: pd.DataFrame
    method: str
    n_duplicated_symbols: int
    n_rows_before: int
    n_rows_after: int
    duplicated_symbols: list[str]


def clean_ensembl_id(identifier: str) -> str:
    """Remove the version suffix from an Ensembl Gene ID.

    Examples:
        >>> clean_ensembl_id("ENSG00000141510.17")
        'ENSG00000141510'
        >>> clean_ensembl_id("TP53")
        'TP53'

    """
    text = str(identifier).strip()
    match = ENSEMBL_GENE_RE.match(text)
    if match:
        return match.group(1).upper()
    return text


def classify_identifier(identifier: str) -> str:
    """Classifica un singolo identificativo genico."""
    text = str(identifier).strip()
    if ENSEMBL_GENE_RE.match(text):
        return ID_TYPE_ENSEMBL
    if ENTREZ_RE.match(text):
        return ID_TYPE_ENTREZ
    if SYMBOL_RE.match(text):
        return ID_TYPE_SYMBOL
    return ID_TYPE_OTHER


def detect_id_type(identifiers: Sequence[str], threshold: float = 0.6) -> str:
    """Determine the prevailing identifier type in a list."""
    if len(identifiers) == 0:
        return ID_TYPE_OTHER
    series = pd.Series([classify_identifier(i) for i in identifiers])
    top = series.value_counts(normalize=True)
    kind, share = top.index[0], top.iloc[0]
    LOGGER.info(
        "Tipo di identificativo genico prevalente: %s (%.1f%% delle righe)",
        kind,
        share * 100,
    )
    if share < threshold:
        LOGGER.warning(
            "Identificativi eterogenei (%s): la conversione potrebbe essere "
            "parziale.",
            dict(top.round(3)),
        )
    return str(kind)


# --------------------------------------------------------------------------- #
# Conversion with mygene
# --------------------------------------------------------------------------- #


def _cache_path(cache_dir: Path, id_type: str) -> Path:
    return Path(cache_dir) / f"mygene_{id_type}.json"


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Cache mygene illeggibile (%s): verra' rigenerata", exc)
        return {}


def _save_cache(path: Path, payload: dict[str, dict[str, str]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError as exc:
        LOGGER.warning("Impossibile salvare la cache mygene in %s: %s", path, exc)


def query_mygene(
    identifiers: Sequence[str],
    id_type: str,
    cache_dir: Path,
    chunk_size: int = 1000,
) -> dict[str, dict[str, str]]:
    """Query mygene.info to convert the IDs into gene symbols.

    The results are stored in ``data/cache``; with no network or no ``mygene``
    available, whatever is in the cache is returned.

    """
    scopes = {
        ID_TYPE_ENSEMBL: "ensembl.gene",
        ID_TYPE_ENTREZ: "entrezgene",
        ID_TYPE_SYMBOL: "symbol,alias",
    }.get(id_type, "symbol,alias,ensembl.gene,entrezgene")

    cache_file = _cache_path(cache_dir, id_type)
    cache = _load_cache(cache_file)
    resolved_in_cache = sum(1 for entry in cache.values() if entry.get("symbol"))
    if cache and resolved_in_cache < 0.05 * len(cache):
        # cache produced by a run in which the service was unreachable
        LOGGER.warning(
            "La cache mygene non contiene conversioni utili (%d/%d): viene "
            "ignorata e l'interrogazione ripetuta.",
            resolved_in_cache,
            len(cache),
        )
        cache = {}
    missing = [i for i in dict.fromkeys(identifiers) if i not in cache]
    if not missing:
        LOGGER.info("Annotazione genica servita interamente dalla cache locale")
        return cache

    try:
        import mygene  # import locale: dipendenza opzionale
    except ImportError:
        LOGGER.error(
            "Libreria 'mygene' non installata: impossibile convertire gli ID in "
            "simboli genici. Installare con 'pip install mygene' oppure usare "
            "--no-mygene."
        )
        return cache

    client = mygene.MyGeneInfo()
    query_failed = False
    LOGGER.info(
        "Interrogazione mygene.info per %d identificativi (%s)...", len(missing), scopes
    )
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start: start + chunk_size]
        try:
            response = client.querymany(
                chunk,
                scopes=scopes,
                fields="symbol,name,entrezgene,ensembl.gene",
                species="human",
                verbose=False,
                returnall=False,
            )
        except Exception as exc:  # errori di rete/servizio/certificati TLS
            query_failed = True
            LOGGER.error(
                "Interrogazione mygene fallita (%s). Verranno usate le fonti "
                "alternative disponibili.",
                exc,
            )
            break
        for hit in response:
            query = str(hit.get("query", ""))
            if hit.get("notfound"):
                cache.setdefault(query, {"symbol": "", "name": "", "status": "not_found"})
                continue
            previous = cache.get(query)
            entry = {
                "symbol": str(hit.get("symbol", "") or ""),
                "name": str(hit.get("name", "") or ""),
                "status": "converted",
            }
            if previous and previous.get("symbol") and previous["symbol"] != entry["symbol"]:
                entry["status"] = "ambiguous_multiple_hits"
                entry["symbol"] = previous["symbol"]  # keep the first hit
            cache[query] = entry
        LOGGER.info(
            "  ... %d/%d identificativi processati", min(start + chunk_size, len(missing)),
            len(missing),
        )

    if query_failed:
        # the query did not complete: negative outcomes are not cached,
        # so that a later run can try again.
        return cache
    for identifier in missing:
        cache.setdefault(identifier, {"symbol": "", "name": "", "status": "not_found"})
    _save_cache(cache_file, cache)
    return cache


def load_ncbi_gene_info(
    cache_dir: Path, timeout: int = 300
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Download (only once) and index NCBI's ``Homo_sapiens.gene_info``.

    Returns:
        ``(map by Ensembl gene ID, map by Entrez gene ID)``; each value holds
        ``symbol`` and ``name``.

    """
    from src import downloader  # import locale: evita dipendenze circolari

    cache_dir = Path(cache_dir)
    local = cache_dir / "Homo_sapiens.gene_info.gz"
    result = downloader.download_file(
        NCBI_GENE_INFO_URL, local, timeout=timeout, retries=2, show_progress=True
    )
    if result.status == "failed" or not local.is_file():
        LOGGER.error(
            "Annotazione di riserva NCBI non disponibile (%s): gli identificativi "
            "resteranno non convertiti.",
            result.message,
        )
        return {}, {}

    try:
        table = pd.read_csv(
            local,
            sep="\t",
            compression="gzip",
            usecols=["GeneID", "Symbol", "dbXrefs", "description"],
            dtype=str,
            na_filter=False,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.error("File gene_info NCBI illeggibile (%s)", exc)
        return {}, {}

    by_ensembl: dict[str, dict[str, str]] = {}
    by_entrez: dict[str, dict[str, str]] = {}
    for gene_id, symbol, xrefs, description in table.itertuples(index=False):
        entry = {"symbol": symbol.upper(), "name": description, "status": "converted"}
        by_entrez[gene_id] = entry
        if "Ensembl:" not in xrefs:
            continue
        for ref in xrefs.split("|"):
            if ref.startswith("Ensembl:"):
                by_ensembl[ref.split(":", 1)[1].strip().upper()] = entry
    LOGGER.info(
        "Annotazione NCBI caricata: %d Ensembl ID e %d Entrez ID",
        len(by_ensembl),
        len(by_entrez),
    )
    return by_ensembl, by_entrez


def annotate_with_ncbi(
    identifiers: Sequence[str], id_type: str, cache_dir: Path
) -> dict[str, dict[str, str]]:
    """Convert the identifiers using NCBI's gene_info file."""
    by_ensembl, by_entrez = load_ncbi_gene_info(cache_dir)
    source = by_ensembl if id_type == ID_TYPE_ENSEMBL else by_entrez
    if not source:
        return {}
    mapping = {i: source[i] for i in identifiers if i in source}
    LOGGER.info(
        "Conversione NCBI: %d/%d identificativi risolti", len(mapping), len(identifiers)
    )
    return mapping


def load_manual_map(path: Path) -> dict[str, dict[str, str]]:
    """Load a manual ``identifier -> symbol`` map provided by the user.

    Useful for identifiers retired in recent Ensembl releases, which no conversion
    service can resolve any more (the GSE101108 count file is annotated on hg19).
    The CSV file must have the ``identifier`` and ``gene_symbol`` columns
    (``gene_name`` optional).

    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        table = pd.read_csv(path, dtype=str, comment="#").fillna("")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("Mappa genica manuale illeggibile (%s): ignorata", exc)
        return {}
    required = {"identifier", "gene_symbol"}
    if not required.issubset(table.columns):
        LOGGER.warning(
            "Mappa genica manuale priva delle colonne %s: ignorata", sorted(required)
        )
        return {}
    mapping = {
        clean_ensembl_id(row["identifier"]).upper(): {
            "symbol": row["gene_symbol"].strip().upper(),
            "name": row.get("gene_name", ""),
            "status": "manual_map",
        }
        for _, row in table.iterrows()
        if row["identifier"].strip() and row["gene_symbol"].strip()
    }
    LOGGER.info("Mappa genica manuale caricata: %d voci da %s", len(mapping), path.name)
    return mapping


def build_annotation_table(
    identifiers: Sequence[str],
    cache_dir: Path,
    use_mygene: bool = True,
    use_ncbi_fallback: bool = True,
    manual_map_path: Path | None = None,
) -> pd.DataFrame:
    """Build the gene-identifier annotation table.

    Columns: ``original_id``, ``clean_id``, ``gene_symbol``, ``gene_name``,
    ``id_type``, ``conversion_status``, ``ambiguity``.

    """
    unique_ids = list(dict.fromkeys(str(i).strip() for i in identifiers))
    id_type = detect_id_type(unique_ids)
    clean = {i: clean_ensembl_id(i) for i in unique_ids}

    mapping: dict[str, dict[str, str]] = {}
    if id_type == ID_TYPE_SYMBOL:
        LOGGER.info(
            "Gli identificativi sono gia' simboli genici: nessuna conversione "
            "necessaria."
        )
    else:
        unique_clean = sorted(set(clean.values()))
        if use_mygene:
            mapping = query_mygene(unique_clean, id_type, cache_dir)
        else:
            LOGGER.warning(
                "Conversione mygene disabilitata (--no-mygene): si usera' "
                "direttamente l'annotazione NCBI."
            )
        resolved = sum(1 for v in mapping.values() if v.get("symbol"))
        if use_ncbi_fallback and resolved < 0.5 * len(unique_clean):
            LOGGER.warning(
                "mygene ha risolto solo %d/%d identificativi: si passa "
                "all'annotazione di riserva NCBI (gene_info).",
                resolved,
                len(unique_clean),
            )
            fallback = annotate_with_ncbi(unique_clean, id_type, cache_dir)
            for key, value in fallback.items():
                if not mapping.get(key, {}).get("symbol"):
                    mapping[key] = value

    if manual_map_path is not None:
        manual = load_manual_map(manual_map_path)
        applied = 0
        for key, value in manual.items():
            if key in clean.values() or key in mapping:
                mapping[key] = value
                applied += 1
        if applied:
            LOGGER.info(
                "Mappa manuale applicata a %d identificativi (prevale sulle "
                "conversioni automatiche).",
                applied,
            )

    rows = []
    for original in unique_ids:
        cleaned = clean[original]
        if id_type == ID_TYPE_SYMBOL:
            symbol, name, status = cleaned.upper(), "", "already_symbol"
        else:
            hit = mapping.get(cleaned, {})
            symbol = (hit.get("symbol") or "").upper()
            name = hit.get("name", "")
            status = hit.get("status", "not_queried")
            if not symbol:
                symbol = cleaned.upper()
                status = status if status != "converted" else "empty_symbol"
        rows.append(
            {
                "original_id": original,
                "clean_id": cleaned,
                "gene_symbol": symbol,
                "gene_name": name,
                "id_type": classify_identifier(original),
                "conversion_status": status,
                "ambiguity": "",
            }
        )

    table = pd.DataFrame(rows)
    # piu' identificativi -> stesso simbolo
    duplicated_symbols = table.loc[
        table.duplicated("gene_symbol", keep=False)
        & (table["gene_symbol"].str.len() > 0),
        "gene_symbol",
    ].unique()
    if len(duplicated_symbols):
        table.loc[
            table["gene_symbol"].isin(duplicated_symbols), "ambiguity"
        ] = "multiple_ids_same_symbol"
        LOGGER.warning(
            "%d simboli genici corrispondono a piu' identificativi originali "
            "(le righe verranno aggregate).",
            len(duplicated_symbols),
        )

    converted = int((table["conversion_status"] == "converted").sum())
    not_found = int((table["conversion_status"] == "not_found").sum())
    LOGGER.info(
        "Annotazione: %d identificativi, %d convertiti, %d non trovati",
        len(table),
        converted,
        not_found,
    )
    return table


# --------------------------------------------------------------------------- #
# Aggregation of duplicates
# --------------------------------------------------------------------------- #


def aggregate_duplicate_genes(
    matrix: pd.DataFrame, symbols: Iterable[str], is_count_like: bool
) -> AggregationResult:
    """Aggregate the rows that share the same gene symbol.

    For raw counts the sum is used (counts are additive); for TPM/FPKM the mean
    (already-normalized measures, not additive without corrections).

    """
    method = "sum" if is_count_like else "mean"
    working = matrix.copy()
    working.index = pd.Index([str(s).upper() for s in symbols], name="gene_symbol")

    n_before = working.shape[0]
    duplicated = working.index[working.index.duplicated(keep=False)].unique().tolist()
    if duplicated:
        aggregated = working.groupby(level=0, sort=False).agg(method)
    else:
        aggregated = working
    LOGGER.info(
        "Aggregazione duplicati per simbolo genico: metodo '%s', %d simboli "
        "duplicati, %d -> %d righe",
        method,
        len(duplicated),
        n_before,
        aggregated.shape[0],
    )
    return AggregationResult(
        matrix=aggregated,
        method=method,
        n_duplicated_symbols=len(duplicated),
        n_rows_before=n_before,
        n_rows_after=aggregated.shape[0],
        duplicated_symbols=sorted(duplicated)[:500],
    )


def extract_genes_of_interest(
    matrix: pd.DataFrame, genes: Sequence[str], annotation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the requested genes from the symbol-indexed matrix.

    Returns:
        ``(gene x sample matrix, gene lookup table)``.

    """
    index_upper = pd.Index([str(i).upper() for i in matrix.index])
    lookup_rows = []
    found_rows: list[str] = []

    for gene in genes:
        symbol = str(gene).strip().upper()
        hits = index_upper == symbol
        n_hits = int(hits.sum())
        row = {
            "gene": symbol,
            "found": n_hits > 0,
            "n_matrix_rows": n_hits,
            "identifier_used": "",
            "n_samples_with_value": 0,
            "n_missing_values": 0,
            "duplicated_rows": n_hits > 1,
        }
        if n_hits:
            values = matrix.loc[hits]
            row["n_samples_with_value"] = int(values.notna().any(axis=0).sum())
            row["n_missing_values"] = int(values.isna().sum().sum())
            original = annotation.loc[
                annotation["gene_symbol"] == symbol, "original_id"
            ].tolist()
            row["identifier_used"] = ", ".join(original[:5]) or symbol
            found_rows.append(symbol)
        lookup_rows.append(row)

    selected = matrix.loc[index_upper.isin(found_rows)].copy()
    selected.index = pd.Index(
        [str(i).upper() for i in selected.index], name="gene_symbol"
    )
    # row order matching the one requested in GENES_OF_INTEREST
    selected = selected.loc[[g for g in found_rows if g in selected.index]]
    lookup = pd.DataFrame(lookup_rows)
    missing = lookup.loc[~lookup["found"], "gene"].tolist()
    if missing:
        LOGGER.warning("Geni richiesti NON trovati nella matrice: %s", ", ".join(missing))
    LOGGER.info("Geni richiesti trovati: %d/%d", len(found_rows), len(genes))
    return selected, lookup
