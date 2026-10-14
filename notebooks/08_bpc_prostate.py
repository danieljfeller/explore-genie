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
# # GENIE BPC Prostate v1.0-public — questions Pr1–Pr3
#
# Source: Jacob's *CBIO thoughts* deck, slides 25–27 (context: Parabilis' ERG degrader). Data: AACR Project GENIE BPC Prostate v1.0-public (Synapse).
#
# Conventions as in `README.md`. **ERG-altered** = any ERG alteration: structural variant / fusion (TMPRSS2-ERG and other partners), non-synonymous
# mutation, amplification or deep deletion. Fusions dominate. **Detection caveat:** DNA panels differ in whether and how they call the TMPRSS2-ERG
# rearrangement, so ERG-fusion frequency is partly a panel artefact. Centres / panels that never report an ERG structural variant are listed and the
# analyses are repeated within panels that do.

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
DS = "bpc_prostate"
F = []

b = cb.BPC("Prostate")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "Prostate"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort and alteration matrix

# %%
S = b.survival_dx()
A = b.alterations()
display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index prostate cancer (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("With OS from diagnosis", int(S.time.notna().sum())),
]))
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel; "
      f"{int((S.entry >= S.time).sum())} patients with sequencing report after last follow-up (dropped from left-truncated models only).")
print(f"Deaths: {int(S.event.sum())} / {len(S)}.  Median OS from dx: {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months.")
display(S.stage_dx.value_counts().rename("stage_dx"))
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind.isin(["mut", "fus", "amp", "homdel"])], top=25, title="Most frequent alterations (all kinds)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
freq.head(20)

# %% [markdown]
# ## Pr1 — Characterise ERG alterations
# Frequency by alteration kind, by panel (detection caveat), fusion partners, and OS of ERG-altered vs not.

# %%
erg_kinds = {k: A.status("ERG", k) for k in ["fus", "mut", "amp", "homdel"]}
erg = A.status("ERG", "alt")
display(pd.Series({k: int(v.sum()) for k, v in erg_kinds.items()} | {"any": int(erg.sum()), "tested": len(erg)}).rename("ERG-altered patients"))
panel = S.index.to_series().map(A.sample_of).map(b.panel_of)
by_panel = (pd.crosstab(panel, erg_kinds["fus"].reindex(S.index).fillna(False)).rename(columns={False: "no ERG fusion", True: "ERG fusion"})
            .reindex(columns=["no ERG fusion", "ERG fusion"], fill_value=0))
by_panel["pct_fusion"] = (100 * by_panel["ERG fusion"] / by_panel.sum(axis=1)).round(1)
display(by_panel.sort_values("pct_fusion", ascending=False))
sv_erg = b.sv_long[b.sv_long.gene == "ERG"]
partners = b.sv_raw
if "Site1_Hugo_Symbol" in partners.columns:
    pe = partners[(partners.Site1_Hugo_Symbol == "ERG") | (partners.Site2_Hugo_Symbol == "ERG")]
    partner = np.where(pe.Site1_Hugo_Symbol == "ERG", pe.Site2_Hugo_Symbol, pe.Site1_Hugo_Symbol)
    display(pd.Series(partner).fillna("intragenic / unknown").value_counts().head(10).rename("ERG fusion partner (samples)"))
detecting_panels = by_panel.index[by_panel["ERG fusion"] > 0]
S["panel"] = panel
S["erg_detecting_panel"] = S.panel.isin(detecting_panels)
d, t1 = cb.compare(S, erg, labels=("ERG not altered", "ERG altered"), entry="entry")
ax, _ = cb.km_plot(d, "group", order=["ERG not altered", "ERG altered"], title="Pr1  OS from diagnosis by ERG alteration (all panels)", xlabel="Months from diagnosis")
cb.save(DS, "Pr1", table=t1, fig=ax.figure, slug="erg_vs_not_all_panels")
display(t1)
d2, t1b = cb.compare(S[S.erg_detecting_panel], erg, labels=("ERG not altered", "ERG altered"), entry="entry", adjust=["stage_dx", "age_dx"])
ax, _ = cb.km_plot(d2, "group", order=["ERG not altered", "ERG altered"], title="Pr1  OS from diagnosis by ERG alteration (panels that detect ERG fusions)", xlabel="Months from diagnosis")
cb.save(DS, "Pr1", table=t1b, fig=ax.figure, slug="erg_vs_not_detecting_panels")
display(t1b)
cb.save(DS, "Pr1", table=by_panel.reset_index(), slug="erg_fusion_by_panel")
cb.finding("Pr1", "descriptive",
           f"ERG altered in {int(erg.sum())}/{len(erg)} patients ({100 * erg.mean():.1f}%; fusions {int(erg_kinds['fus'].sum())}, mutations {int(erg_kinds['mut'].sum())}, amp {int(erg_kinds['amp'].sum())}). "
           f"Fusion frequency ranges {by_panel.pct_fusion.min()}–{by_panel.pct_fusion.max()}% across panels (detection artefact). "
           f"OS, detecting panels only: median {t1b.median_months[1]} vs {t1b.median_months[0]} mo, p={cb.fmt_p(t1b.p_logrank[0])}, adjusted HR (LT) {t1b.attrs['HR']:.2f}.", F)

# %% [markdown]
# ## Pr2 (replicate) — Among ERG-altered patients, AR vs PTEN co-alteration shows no significant OS difference
# Within ERG-altered patients: AR co-alteration (amplification or mutation) vs PTEN co-alteration (deep deletion or mutation) vs neither (slides 26–27).
# cBioPortal's comparison dropped patients with both; here they are their own group.

# %%
erg_ids = erg.index[erg]
ar = (A.status("AR", "amp") | A.status("AR", "mut")).rename("AR")
pten = (A.status("PTEN", "homdel") | A.status("PTEN", "mut")).rename("PTEN")
st = pd.concat([ar, pten], axis=1, join="inner").loc[lambda x: x.index.isin(erg_ids)]
st["profile"] = np.select([st.AR & st.PTEN, st.AR, st.PTEN], ["AR + PTEN", "AR co-altered", "PTEN co-altered"], default="neither")
order2 = ["neither", "AR co-altered", "PTEN co-altered", "AR + PTEN"]
d = S.join(st.profile, how="inner")
display(d.profile.value_counts().rename("ERG-altered patients"))
ax, t2 = cb.km_plot(d, "profile", order=order2, title="Pr2  ERG-altered: OS from diagnosis by AR / PTEN co-alteration", xlabel="Months from diagnosis")
cb.save(DS, "Pr2", table=t2, fig=ax.figure, slug="ar_vs_pten_in_erg")
display(t2)
dx = d[d.profile.isin(["AR co-altered", "PTEN co-altered"])]
t2x = cb.km_table(dx, "profile", order=["AR co-altered", "PTEN co-altered"])
h2 = cb.hr(dx.assign(feat=(dx.profile == "PTEN co-altered")), "feat", entry="entry")
t2x["HR_LT_pten_vs_ar"] = h2["HR"]
cb.save(DS, "Pr2", table=t2x, slug="ar_vs_pten_two_groups")
display(t2x)
# AR amplification is a marker of castration resistance (acquired under ADT): sequencing after progression inflates its apparent effect
cb.finding("Pr2", "replicated" if t2x.p_logrank[0] > 0.05 else "not replicated",
           f"ERG-altered: AR co-altered n={t2x.n[0]} median {t2x.median_months[0]} vs PTEN co-altered n={t2x.n[1]} median {t2x.median_months[1]} mo, "
           f"log-rank p={cb.fmt_p(t2x.p_logrank[0])}, HR (LT) PTEN vs AR {h2['HR']:.2f}. Four-group log-rank p={cb.fmt_p(t2.p_logrank[0])}. "
           f"AR amplification marks castration resistance acquired under therapy, so the AR group is enriched for late-sequenced patients.", F)

# %% [markdown]
# ## Pr3 — Mutational profiles that convey higher risk in ERG-altered patients
# Systematic screen of every co-alteration within ERG-altered patients (≥ 10 carriers / non-carriers), log-rank, left-truncated Cox HR, BH FDR;
# repeated in panels that detect ERG fusions. Then the same screen in ERG-wild-type patients, to separate ERG-specific from general prognostic markers.

# %%
scr = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel", "fus"), min_n=10, entry="entry", patients=erg_ids)
scr = scr[scr.gene != "ERG"].reset_index(drop=True)
scr_wt = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel", "fus"), min_n=10, entry="entry", patients=erg.index[~erg])
scr_wt = scr_wt[scr_wt.gene != "ERG"].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
cb.forest_plot(scr, top=15, title="Pr3  Within ERG-altered: co-alterations vs OS", ax=axes[0])
cb.forest_plot(scr_wt, top=15, title="Pr3  Within ERG-wild-type (comparison)", ax=axes[1])
fig.tight_layout()
cb.save(DS, "Pr3", table=scr, fig=fig, slug="screen_in_erg_altered")
cb.save(DS, "Pr3", table=scr_wt, slug="screen_in_erg_wt")
both = scr.merge(scr_wt[["feature", "HR", "p_logrank", "q_bh"]], on="feature", how="left", suffixes=("_erg", "_wt"))
display(both.head(15)[["feature", "n_alt", "median_alt", "median_wt", "HR_erg", "p_logrank_erg", "q_bh_erg", "HR_wt", "p_logrank_wt", "q_bh_wt"]])
hits = scr[scr.q_bh < 0.1]
for r in hits.head(2).itertuples():
    d, t = cb.compare(S.loc[erg_ids], A.status(r.gene, r.kind), labels=(f"{r.feature} absent", f"{r.feature} present"), entry="entry")
    ax, _ = cb.km_plot(d, "group", order=[f"{r.feature} absent", f"{r.feature} present"], title=f"Pr3  ERG-altered: OS by {r.feature}", xlabel="Months from diagnosis")
    cb.save(DS, "Pr3", table=t, fig=ax.figure, slug=f"erg_by_{r.gene}_{r.kind}")
cb.finding("Pr3", "signal" if len(hits) else "no signal at FDR<0.1",
           f"ERG-altered n={len(erg_ids)}, {len(scr)} features tested. FDR<0.1: " +
           (", ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()) or f"none (best {scr.feature[0]} HR {scr.HR[0]:.2f} p={cb.fmt_p(scr.p_logrank[0])})") +
           ". ERG-specific if absent from the ERG-wt screen: " +
           (", ".join(r.feature for r in both[(both.q_bh_erg < 0.1) & ((both.q_bh_wt >= 0.1) | both.q_bh_wt.isna())].itertuples()) or "none") + ".", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
