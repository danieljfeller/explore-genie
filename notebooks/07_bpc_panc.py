# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GENIE BPC PANC v1.0-public — question P1
#
# Source: Jacob's *CBIO thoughts* deck, slide 24. Data: AACR Project GENIE BPC PANC v1.0-public (Synapse).
#
# Conventions as in `README.md`: index date = diagnosis of the index pancreatic cancer, OS in months; *X mut* = non-synonymous mutation;
# `amp` / `homdel` / `fus` separate; patients compared for a gene only if their panel covers it; KM / log-rank as cBioPortal, Cox HR
# left-truncated at the sequencing report (*HR (LT)*). Pancreatic cancer is sequenced late and dies fast, so left truncation matters more here than anywhere else in the deck.

# %%
import sys, re, warnings
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "cbio_lib.py").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cbio_lib as cb

cb.setup_plots()
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.3g}".format
pd.options.display.width = 200
DS = "bpc_panc"
F = []

b = cb.BPC("PANC")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "PANC"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort and alteration matrix

# %%
S = b.survival_dx()
A = b.alterations()
display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index pancreatic cancer (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("With OS from diagnosis", int(S.time.notna().sum())),
]))
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel; "
      f"{int((S.entry >= S.time).sum())} patients with sequencing report after last follow-up (dropped from left-truncated models only).")
print(f"Deaths: {int(S.event.sum())} / {len(S)}.  Median OS from dx: {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months; "
      f"median months from diagnosis to sequencing report: {S.entry.median():.1f}.")
display(S.stage_dx.value_counts().rename("stage_dx"))
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes (non-synonymous)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
freq.head(15)

# %% [markdown]
# ## P1 — How is OS of KRAS-mutant patients further differentiated by co-mutations? Which co-mutations associate with worse outcome?
# Within KRAS-mutant patients: every co-alteration (mutation, amplification, deep deletion) with ≥ 10 carriers and non-carriers; log-rank,
# left-truncated Cox HR, BH FDR. CDKN2A / CDKN2B / MTAP deep deletions are of special interest and are shown individually, as is the
# KRAS-wild-type group for reference.

# %%
kras = A.status("KRAS")
kras_ids = kras.index[kras]
d, t0 = cb.compare(S, kras, labels=("KRAS wt", "KRAS mut"), entry="entry")
ax, _ = cb.km_plot(d, "group", order=["KRAS wt", "KRAS mut"], title="P1  OS from diagnosis by KRAS mutation status (reference)", xlabel="Months from diagnosis")
cb.save(DS, "P1", table=t0, fig=ax.figure, slug="kras_vs_wt")
display(t0)
scr = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel"), min_n=10, entry="entry", patients=kras_ids)
scr = scr[scr.gene != "KRAS"].reset_index(drop=True)
ax = cb.forest_plot(scr, top=15, title="P1  Co-alterations and OS within KRAS-mutant pancreatic cancer (top 15)")
cb.save(DS, "P1", table=scr, fig=ax.figure, slug="comutation_screen_in_kras")
display(scr.head(15))

# %%
special = [("CDKN2A", "homdel"), ("CDKN2B", "homdel"), ("MTAP", "homdel"), ("CDKN2A", "mut"), ("TP53", "mut"), ("SMAD4", "mut"), ("SMAD4", "homdel")]
rows = []
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for (g, k), ax in zip(special[:4], axes.ravel()):
    d, t = cb.compare(S.loc[kras_ids], A.status(g, k), labels=(f"{g} {k} absent", f"{g} {k} present"), entry="entry")
    cb.km_plot(d, "group", order=[f"{g} {k} absent", f"{g} {k} present"], ax=ax, risk_table=False, title=f"P1  KRAS-mut: OS by {g} {k}", xlabel="Months from diagnosis")
    rows.append(t.assign(feature=f"{g} {k}"))
for g, k in special[4:]:
    d, t = cb.compare(S.loc[kras_ids], A.status(g, k), labels=(f"{g} {k} absent", f"{g} {k} present"), entry="entry")
    rows.append(t.assign(feature=f"{g} {k}"))
fig.tight_layout()
sp = pd.concat(rows).set_index(["feature", "group"])
cb.save(DS, "P1", table=sp.reset_index(), fig=fig, slug="special_genes")
display(sp)
# CDKN2A/B + MTAP co-deletion
codel = (A.status("CDKN2A", "homdel") & A.status("MTAP", "homdel")).rename("codel")
d, tcd = cb.compare(S.loc[kras_ids], codel, labels=("not co-deleted", "CDKN2A + MTAP co-deleted"), entry="entry", adjust=["stage_dx", "age_dx"])
ax, _ = cb.km_plot(d, "group", order=["not co-deleted", "CDKN2A + MTAP co-deleted"], title="P1  KRAS-mut: OS by CDKN2A + MTAP co-deletion", xlabel="Months from diagnosis")
cb.save(DS, "P1", table=tcd, fig=ax.figure, slug="cdkn2a_mtap_codeletion")
display(tcd)
hits = scr[scr.q_bh < 0.1]
cd = sp.loc[("CDKN2A homdel", "CDKN2A homdel present")]; mt = sp.loc[("MTAP homdel", "MTAP homdel present")]
cb.finding("P1", "signal" if len(hits) else "no signal at FDR<0.1",
           f"KRAS-mut n={len(kras_ids)}. FDR<0.1 co-alterations: " + (", ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()) or "none") +
           f". CDKN2A homdel n={cd.n}: median {cd.median_months} vs {sp.loc[('CDKN2A homdel', 'CDKN2A homdel absent'), 'median_months']} mo, HR (LT) {cd.HR:.2f} (p={cb.fmt_p(cd.p_cox)}); "
           f"MTAP homdel n={mt.n}: HR {mt.HR:.2f} (p={cb.fmt_p(mt.p_cox)}); co-deletion adjusted HR {tcd.attrs['HR']:.2f} (p={cb.fmt_p(tcd.attrs['p_cox'])}).", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
