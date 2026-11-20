# -*- coding: utf-8 -*-
"""Benjamini-Hochberg correction of the KM-plotter subgroup estimates (Table 5).

KM-plotter is a web tool, so its univariate hazard ratios cannot be recomputed
here; they are stored as reported in Table 5 in ``external/kmplotter_table5.csv``
(probe 216119_s_at, median split). This script applies a post hoc BH correction
across every subgroup estimate that has a log-rank p value, as done in the
revision, and marks with a dagger the estimates that stay at q < 0.05.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "external" / "kmplotter_table5.csv"
TAB = ROOT / "results" / "tables"


def bh(pvals):
    """Benjamini-Hochberg FDR."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    ranked = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.clip(q, 0, 1)
    return out


def main() -> None:
    km = pd.read_csv(SRC)
    tested = km["p_logrank"].notna()
    km.loc[tested, "q_BH"] = bh(km.loc[tested, "p_logrank"])
    km["significant_after_BH"] = km["q_BH"] < 0.05
    km["mark"] = np.where(km["significant_after_BH"], "†", "")

    TAB.mkdir(parents=True, exist_ok=True)
    km.to_csv(TAB / "kmplotter_table5_BH.csv", index=False)

    nominal = km[tested & (km["p_logrank"] < 0.05)]
    summary = (
        f"{int(tested.sum())} subgroup estimates with a p value; "
        f"{len(nominal)} nominally significant ({int((nominal.hr < 1).sum())} with HR < 1, "
        f"{int((nominal.hr > 1).sum())} with HR > 1); "
        f"{int(km['significant_after_BH'].sum())} with q < 0.05 after Benjamini-Hochberg"
    )
    print(summary)
    print(km.loc[km["significant_after_BH"],
                 ["variable", "subgroup", "endpoint", "hr", "p_logrank", "q_BH"]].to_string(index=False))
    (TAB / "kmplotter_table5_BH_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
