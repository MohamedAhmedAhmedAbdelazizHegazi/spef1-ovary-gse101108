# -*- coding: utf-8 -*-
"""Combined histotype figure (HGSC/LGSC split) + dataset clinical table + TMA table."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent / "GSE101108_pipeline"
# The tissue-microarray scores are not distributed with this repository (they are
# the immunohistochemistry raw data of the paper). Place the Excel file at
# <repo>/data/TMA_SPEF1_expression.xlsx before running the TMA part of this script.
TMA_FILE = ROOT.parent / "data" / "TMA_SPEF1_expression.xlsx"
FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"


def bh(pvals):
    """Benjamini-Hochberg FDR."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    ranked = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.clip(q, 0, 1)
    return out


def star(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def pairwise_positivity(labels, groups, pos):
    """Fisher exact (pos vs neg) over all group pairs, BH-corrected."""
    import pandas as _pd
    from itertools import combinations
    df = _pd.DataFrame({"g": groups, "pos": pos})
    res = []
    for g1, g2 in combinations(labels, 2):
        a, b_ = df[df.g == g1], df[df.g == g2]
        if len(a) == 0 or len(b_) == 0:
            continue
        table = [[int(a.pos.sum()), int((~a.pos).sum())],
                 [int(b_.pos.sum()), int((~b_.pos).sum())]]
        _, p = stats.fisher_exact(table)
        res.append([g1, g2, p])
    if not res:
        return _pd.DataFrame(columns=["group_1", "group_2", "p", "q"])
    r = _pd.DataFrame(res, columns=["group_1", "group_2", "p"])
    r["q"] = bh(r["p"].values)
    return r

ORDER = ["HGSC", "LGSC", "Clear cell", "Endometrioid", "Mucinous"]
HCOL = {"HGSC": "#4C78A8", "LGSC": "#72B7B2", "Clear cell": "#F58518",
        "Endometrioid": "#54A24B", "Mucinous": "#E45756"}
NEUTRAL = "#6E7B8B"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------- dati dataset ----------
counts = pd.read_csv(PIPE / "data/processed/GSE101108_complete_expression_matrix.csv.gz", index_col=0)
counts.columns = [c.split("|")[0] for c in counts.columns]
meta = pd.read_excel(PIPE / "data/metadata/GSE101108_metadata_normalized.xlsx")
hist = pd.read_excel(PIPE / "data/metadata/GSE101108_histotype_classification.xlsx", sheet_name="classificazione")
FINE = {"HGSC": "HGSC", "LGSC": "LGSC", "CCC": "Clear cell", "EC": "Endometrioid", "MC": "Mucinous"}
hist["fine"] = hist["histotype_original"].map(FINE)
md = meta.merge(hist[["gsm_id", "fine"]], on="gsm_id").dropna(subset=["fine"]).set_index("gsm_id")
samples = [c for c in counts.columns if c in md.index]
md = md.loc[samples]
raw = counts.loc["SPEF1", samples]
m = counts[samples].to_numpy(float); usable = (m > 0).all(1)
sf = np.exp(np.median(np.log(m[usable]) - np.log(m[usable]).mean(1, keepdims=True), 0))
logs = np.log2(counts.loc["SPEF1", samples] / sf + 1.0)
md["log2"] = logs.values; md["raw"] = raw.values; md["pos"] = raw.values >= 5
md["age"] = pd.to_numeric(md["age"], errors="coerce")
md["stage"] = md["stage"].astype(str).str.upper().str.strip()

# ---------- figura combinata (5 gruppi) ----------
kw = stats.kruskal(*[md.loc[md.fine == h, "log2"] for h in ORDER if (md.fine == h).sum() > 0])
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
a = ax[0, 0]
data = [md.loc[md.fine == h, "log2"].values for h in ORDER]
bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.62)
for box, h in zip(bp["boxes"], ORDER): box.set(facecolor=HCOL[h], alpha=0.45, edgecolor=HCOL[h])
for mm in bp["medians"]: mm.set_color("black")
rng = np.random.default_rng(1)
for i, (v, h) in enumerate(zip(data, ORDER), 1):
    a.scatter(np.full(len(v), i) + rng.normal(0, 0.06, len(v)), v, s=20, color=HCOL[h],
              edgecolor="white", linewidth=0.4, zorder=3)
a.set_xticks(range(1, 6)); a.set_xticklabels([f"{h}\n(n={len(d)})" for h, d in zip(ORDER, data)], fontsize=9)
a.set_ylabel("SPEF1, log2(normalized counts + 1)")
a.set_title(f"A   SPEF1 mRNA by histotype (Kruskal–Wallis p = {kw.pvalue:.3f})", loc="left")
# pairwise stars (Mann-Whitney, BH) only for significant pairs
from itertools import combinations
pw = []
for h1, h2 in combinations(ORDER, 2):
    d1, d2 = md.loc[md.fine == h1, "log2"], md.loc[md.fine == h2, "log2"]
    if len(d1) >= 1 and len(d2) >= 1:
        pw.append([h1, h2, stats.mannwhitneyu(d1, d2).pvalue])
pw = pd.DataFrame(pw, columns=["g1", "g2", "p"]); pw["q"] = bh(pw["p"].values)
top = max((v.max() for v in data if len(v)), default=1); lvl = top + 0.35
for _, r in pw[pw.q < 0.05].sort_values("q").iterrows():
    x1, x2 = ORDER.index(r.g1) + 1, ORDER.index(r.g2) + 1
    a.plot([x1, x1, x2, x2], [lvl, lvl + 0.08, lvl + 0.08, lvl], color="black", lw=1)
    a.text((x1 + x2) / 2, lvl + 0.10, star(r.q), ha="center", va="bottom", fontsize=13)
    lvl += 0.55
print("\n=== PAIRWISE SPEF1 mRNA (Mann-Whitney, BH) ===")
print(pw.sort_values("q").to_string(index=False))
b = ax[0, 1]
det = md.groupby("fine")["pos"].agg(["sum", "size"]).reindex(ORDER)
det["pct"] = 100 * det["sum"] / det["size"]
bars = b.bar(range(5), det["pct"], color=[HCOL[h] for h in ORDER], alpha=0.85)
for rect, (_, r) in zip(bars, det.iterrows()):
    b.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5, f"{int(r['sum'])}/{int(r['size'])}", ha="center", fontsize=10)
b.set_xticks(range(5)); b.set_xticklabels(ORDER, fontsize=9, rotation=15, ha="right")
b.set_ylabel("SPEF1-positive samples (%)"); b.set_ylim(0, 105)
b.set_title("B   Detection rate (≥ 5 raw counts)", loc="left")
c = ax[1, 0]
stages = ["I", "II"]
sd = [md.loc[md.stage == s, "log2"].values for s in stages]
bp = c.boxplot(sd, patch_artist=True, showfliers=False, widths=0.5)
for box in bp["boxes"]: box.set(facecolor=NEUTRAL, alpha=0.4)
for mm in bp["medians"]: mm.set_color("black")
for i, v in enumerate(sd, 1):
    c.scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v, s=20, color=NEUTRAL, edgecolor="white", linewidth=0.4, zorder=3)
pstage = stats.mannwhitneyu(sd[0], sd[1]).pvalue
c.set_xticks([1, 2]); c.set_xticklabels([f"FIGO {s}\n(n={len(v)})" for s, v in zip(stages, sd)])
c.set_ylabel("SPEF1, log2(normalized counts + 1)")
c.set_title(f"C   SPEF1 by FIGO stage (p = {pstage:.3f})", loc="left")
dd = ax[1, 1]
dd.scatter(md["age"], md["log2"], s=26, color=[HCOL.get(h, NEUTRAL) for h in md.fine], edgecolor="white", linewidth=0.4)
v = md.dropna(subset=["age", "log2"]); sl, ic = np.polyfit(v.age, v.log2, 1)
xs = np.linspace(v.age.min(), v.age.max(), 50); dd.plot(xs, sl * xs + ic, "--", color="black", lw=1)
rho_age, p_age = stats.spearmanr(v.age, v.log2)
dd.set_xlabel("Age at diagnosis (years)"); dd.set_ylabel("SPEF1, log2(normalized counts + 1)")
dd.set_title(f"D   SPEF1 vs age (Spearman ρ = {rho_age:.2f}, p = {p_age:.3f})", loc="left")
dd.legend(handles=[Line2D([], [], marker="o", ls="", color=HCOL[h], label=h, ms=8) for h in ORDER], fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "Figure_SPEF1_hist_clinical_combined.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "Figure_SPEF1_hist_clinical_combined.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("Combined figure (HGSC/LGSC split) regenerated. Kruskal-Wallis p =", round(kw.pvalue, 4))

# ---------- dataset clinical table ----------
rows = []
for h in ORDER + ["All"]:
    s = md if h == "All" else md[md.fine == h]
    rows.append({
        "Histotype": h if h != "All" else "All histotypes",
        "n": len(s),
        "Age, median (range)": f"{int(s.age.median())} ({int(s.age.min())}–{int(s.age.max())})",
        "FIGO I / II": f"{(s.stage=='I').sum()} / {(s.stage=='II').sum()}",
        "SPEF1-positive, n (%)": f"{int(s.pos.sum())} ({100*s.pos.mean():.1f}%)",
        "SPEF1, median log2 (IQR)": f"{s.log2.median():.2f} ({s.log2.quantile(.25):.2f}–{s.log2.quantile(.75):.2f})",
    })
clin = pd.DataFrame(rows); clin.to_csv(TAB / "dataset_clinical_table.csv", index=False)
print("\n=== DATASET CLINICAL TABLE ===\n", clin.to_string(index=False))

# pairwise positivity (Fisher, BH) between histotypes in the dataset
pwc = pairwise_positivity(ORDER, md.fine.values, md.pos.values)
pwc.to_csv(TAB / "dataset_clinical_pairwise.csv", index=False)
sig_c = pwc[pwc.q < 0.05]
note_c = ("; ".join(f"{r.group_1} vs {r.group_2}, p={r.p:.3f} (q={r.q:.3f})" for _, r in sig_c.iterrows())
          if len(sig_c) else "no pairwise comparison reached significance after Benjamini–Hochberg correction")
print("\n=== DATASET PAIRWISE POSITIVITY (Fisher, BH) ===")
print(pwc.sort_values("q").to_string(index=False))
print("NOTE:", note_c)
open(TAB / "dataset_clinical_note.txt", "w", encoding="utf-8").write(note_c)

# ---------- TMA table (only if the tissue-microarray file is present) ----------
if not TMA_FILE.exists():
    print(f"\nTMA file not found at {TMA_FILE}; skipping Table 6.")
    print("Figure 3 and the dataset clinical table (Table 2) were written.")
    raise SystemExit(0)
tmaraw = pd.read_excel(TMA_FILE, header=None)
t = tmaraw.iloc[11:].copy(); t.columns = list(tmaraw.iloc[10])
t = t.dropna(subset=["Pathology diagnosis"]).reset_index(drop=True)
def histo(x):
    x = str(x).lower()
    for k, v in [("high grade serous", "HGSC"), ("low grade serous", "LGSC"), ("clear cell", "Clear cell"),
                 ("endometrioid", "Endometrioid"), ("mucinous", "Mucinous"),
                 ("metasta", "LN metastasis"), ("normal", "Normal"), ("adjacent", "Normal")]:
        if k in x: return v
    return str(x)[:20]
t["H"] = t["Pathology diagnosis"].map(histo)
t["S"] = t["SPEF1 expression"].astype(str).str.strip()
t.loc[t.H == "Normal", "S"] = "-"          # all normal cores are negative
t["pos"] = t["S"].isin(["+", "++", "+++"])
TORD = ["HGSC", "LGSC", "Clear cell", "Endometrioid", "Mucinous", "LN metastasis", "Normal"]
rows = []
for h in TORD:
    s = t[t.H == h]
    if len(s) == 0: continue
    rows.append({
        "Histotype": h, "n": len(s),
        "SPEF1-positive, n (%)": f"{int(s.pos.sum())} ({100*s.pos.mean():.1f}%)",
        "–": int((s.S == "-").sum()), "+": int((s.S == "+").sum()),
        "++": int((s.S == "++").sum()), "+++": int((s.S == "+++").sum()),
    })
tma = pd.DataFrame(rows); tma.to_csv(TAB / "tma_table.csv", index=False)
print("\n=== TMA TABLE (semi-quantitative) ===\n", tma.to_string(index=False))

# pairwise positivity (Fisher, BH) between histotypes in the TMA (carcinomas + normal as reference)
TMA_MAIN = ["HGSC", "LGSC", "Clear cell", "Endometrioid", "Mucinous", "Normal"]
tt = t[t.H.isin(TMA_MAIN)]
pwt = pairwise_positivity(TMA_MAIN, tt.H.values, tt.pos.values)
pwt.to_csv(TAB / "tma_pairwise.csv", index=False)
sig_t = pwt[pwt.q < 0.05]
note_t = ("; ".join(f"{r.group_1} vs {r.group_2}, p={r.p:.3f} (q={r.q:.3f})" for _, r in sig_t.iterrows())
          if len(sig_t) else "no pairwise comparison reached significance after Benjamini–Hochberg correction")
print("\n=== TMA PAIRWISE POSITIVITY (Fisher, BH) ===")
print(pwt.sort_values("q").to_string(index=False))
print("NOTE:", note_t)
open(TAB / "tma_note.txt", "w", encoding="utf-8").write(note_t)
