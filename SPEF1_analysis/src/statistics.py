"""Shared statistical functions (non-parametric tests, FDR, effect size).

Every analysis uses non-parametric tests: the RNA-seq counts of a lowly
expressed gene like SPEF1 do not meet the assumptions of parametric tests and
the per-histotype sample size is strongly imbalanced.

"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    NaNs are propagated without consuming a test.

    Examples:
        >>> np.round(benjamini_hochberg([0.01, 0.02, 0.03]), 4)
        array([0.03, 0.03, 0.03])

    """
    values = np.asarray(pvalues, dtype=float)
    qvalues = np.full(values.shape, np.nan)
    valid = ~np.isnan(values)
    if not valid.any():
        return qvalues

    subset = values[valid]
    n = subset.size
    order = np.argsort(subset)
    ranked = subset[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0, 1)
    qvalues[valid] = result
    return qvalues


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Effect size di Cliff (non parametrico), in [-1, 1].

    ``|delta|``: <0.15 trascurabile, <0.33 piccolo, <0.47 medio, oltre grande.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (a.size * b.size))


def effect_size_label(delta: float) -> str:
    """Etichetta qualitativa dell'effect size di Cliff."""
    if np.isnan(delta):
        return "n/d"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "trascurabile"
    if magnitude < 0.33:
        return "piccolo"
    if magnitude < 0.474:
        return "medio"
    return "grande"


def kruskal_wallis(groups: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Test di Kruskal-Wallis.

    Returns:
        ``(H, p, epsilon^2)`` dove epsilon^2 e' l'effect size.
    """
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[~np.isnan(g)] for g in clean]
    clean = [g for g in clean if g.size > 0]
    if len(clean) < 2:
        return np.nan, np.nan, np.nan
    statistic, pvalue = stats.kruskal(*clean)
    n = sum(g.size for g in clean)
    epsilon_squared = (statistic - len(clean) + 1) / (n - len(clean)) if n > len(clean) else np.nan
    return float(statistic), float(pvalue), float(epsilon_squared)


def pairwise_mannwhitney(
    values: pd.Series, groups: pd.Series, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """Pairwise comparisons with Mann-Whitney and BH correction.

    Args:
        values: numeric values (indexed by sample).
        groups: group labels with the same index.
        order: order of the groups to compare.

    Returns:
        Table with medians, U, p, q (BH) and Cliff's delta for each pair.

    """
    labels = list(order) if order is not None else sorted(groups.dropna().unique())
    labels = [label for label in labels if (groups == label).sum() > 0]

    rows = []
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            a = values[groups == first].dropna()
            b = values[groups == second].dropna()
            if a.size < 2 or b.size < 2:
                pvalue, statistic = np.nan, np.nan
            else:
                statistic, pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")
            delta = cliffs_delta(a, b)
            rows.append(
                {
                    "group_1": first,
                    "group_2": second,
                    "n_1": int(a.size),
                    "n_2": int(b.size),
                    "median_1": float(a.median()) if a.size else np.nan,
                    "median_2": float(b.median()) if b.size else np.nan,
                    "U": statistic,
                    "p_value": pvalue,
                    "cliffs_delta": delta,
                    "effect_size": effect_size_label(delta),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q_value_BH"] = benjamini_hochberg(table["p_value"])
        table["significant"] = table["q_value_BH"] < 0.05
    return table


def pairwise_fisher(
    positive: pd.Series, groups: pd.Series, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """Pairwise comparisons of positive proportions (Fisher exact test)."""
    labels = list(order) if order is not None else sorted(groups.dropna().unique())
    labels = [label for label in labels if (groups == label).sum() > 0]

    rows = []
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            a = positive[groups == first].astype(bool)
            b = positive[groups == second].astype(bool)
            table = np.array(
                [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
            )
            odds, pvalue = stats.fisher_exact(table, alternative="two-sided")
            rows.append(
                {
                    "group_1": first,
                    "group_2": second,
                    "positive_1": int(a.sum()),
                    "n_1": int(a.size),
                    "rate_1": float(a.mean()) if a.size else np.nan,
                    "positive_2": int(b.sum()),
                    "n_2": int(b.size),
                    "rate_2": float(b.mean()) if b.size else np.nan,
                    "odds_ratio": float(odds),
                    "p_value": float(pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_value_BH"] = benjamini_hochberg(frame["p_value"])
        frame["significant"] = frame["q_value_BH"] < 0.05
    return frame


def spearman_with_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    min_distinct: int = 4,
) -> dict[str, float]:
    """Spearman correlation with a 95% bootstrap confidence interval.

    Args:
        min_distinct: minimum number of distinct values required in each variable.
            With fewer distinct values (e.g. a gene at zero in almost every sample)
            the coefficient is dominated by ties and can take extreme, meaningless
            values: in those cases NaN is returned instead of a misleading rho.

    """
    a = pd.Series(x, dtype=float).to_numpy()
    b = pd.Series(y, dtype=float).to_numpy()
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    empty = {"rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "n": int(a.size)}
    if a.size < 5:
        return empty
    if np.unique(a).size < min_distinct or np.unique(b).size < min_distinct:
        # constant or near-constant gene: correlation dominated by ties
        return empty

    rho, pvalue = stats.spearmanr(a, b)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        if np.all(a[idx] == a[idx][0]) or np.all(b[idx] == b[idx][0]):
            samples[i] = np.nan
            continue
        samples[i] = stats.spearmanr(a[idx], b[idx]).statistic
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {
        "rho": float(rho),
        "p_value": float(pvalue),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(a.size),
    }


def vectorized_spearman(matrix: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Spearman correlation between every row of the matrix and a vector.

    Implemented as a Pearson correlation on the ranks: this makes it possible to
    process tens of thousands of genes in a few seconds.

    Returns:
        Table with ``rho``, ``p_value`` and ``q_value_BH`` for each gene.

    """
    aligned = matrix.loc[:, reference.index]
    ranks = aligned.rank(axis=1).to_numpy(dtype=float)
    ref_ranks = reference.rank().to_numpy(dtype=float)

    ranks_centered = ranks - ranks.mean(axis=1, keepdims=True)
    ref_centered = ref_ranks - ref_ranks.mean()
    numerator = ranks_centered @ ref_centered
    denominator = np.sqrt((ranks_centered**2).sum(axis=1) * (ref_centered**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denominator > 0, numerator / denominator, np.nan)

    n = aligned.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = rho * np.sqrt((n - 2) / np.clip(1 - rho**2, 1e-12, None))
    pvalues = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)
    pvalues = np.where(np.isnan(rho), np.nan, pvalues)

    table = pd.DataFrame(
        {"gene": aligned.index, "rho": rho, "p_value": pvalues}
    ).set_index("gene")
    table["q_value_BH"] = benjamini_hochberg(table["p_value"].to_numpy())
    return table.sort_values("rho", ascending=False)


def vectorized_mannwhitney(
    matrix: pd.DataFrame, group_a: Sequence[str], group_b: Sequence[str]
) -> pd.DataFrame:
    """Row-by-row Mann-Whitney between two sample groups.

    Returns:
        Table with medians, log2 fold-change, p, q (BH) and Cliff's delta.

    """
    a = matrix[list(group_a)].to_numpy(dtype=float)
    b = matrix[list(group_b)].to_numpy(dtype=float)
    statistic, pvalues = stats.mannwhitneyu(a, b, axis=1, alternative="two-sided")

    median_a = np.median(a, axis=1)
    median_b = np.median(b, axis=1)
    mean_a = a.mean(axis=1)
    mean_b = b.mean(axis=1)
    # the values are already log2(normalized + 1): the difference is the log2FC
    log2fc = mean_a - mean_b
    delta = (statistic / (a.shape[1] * b.shape[1])) * 2 - 1  # Cliff's delta da U

    table = pd.DataFrame(
        {
            "gene": matrix.index,
            "median_group_a": median_a,
            "median_group_b": median_b,
            "mean_group_a": mean_a,
            "mean_group_b": mean_b,
            "log2FC": log2fc,
            "cliffs_delta": delta,
            "U": statistic,
            "p_value": pvalues,
        }
    ).set_index("gene")
    table["q_value_BH"] = benjamini_hochberg(table["p_value"].to_numpy())
    return table.sort_values("p_value")


def describe_by_group(
    values: pd.Series, groups: pd.Series, order: Sequence[str] | None = None
) -> pd.DataFrame:
    """Descriptive statistics per group (n, median, IQR, mean, range)."""
    labels = list(order) if order is not None else sorted(groups.dropna().unique())
    rows = []
    for label in labels:
        subset = values[groups == label].dropna()
        if subset.empty:
            continue
        rows.append(
            {
                "group": label,
                "n": int(subset.size),
                "median": float(subset.median()),
                "q1": float(subset.quantile(0.25)),
                "q3": float(subset.quantile(0.75)),
                "mean": float(subset.mean()),
                "sd": float(subset.std(ddof=1)) if subset.size > 1 else np.nan,
                "min": float(subset.min()),
                "max": float(subset.max()),
            }
        )
    return pd.DataFrame(rows)


def format_p(pvalue: float) -> str:
    """Formatting of the p-values for the manuscript text."""
    if pvalue is None or (isinstance(pvalue, float) and np.isnan(pvalue)):
        return "n/d"
    if pvalue < 0.001:
        return f"{pvalue:.1e}".replace("e-0", "e-")
    return f"{pvalue:.3f}"
