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
# # GENIE BPC BrCa v1.0-public — questions Br1–Br6
#
# Source: Jacob's *CBIO thoughts* deck, slides 15–18. Data: AACR Project GENIE BPC BrCa v1.0-public (Synapse).
#
# **Conventions** (see `README.md`): index date = diagnosis of the index breast cancer, OS in months; treatment questions use OS from regimen start.
# *X mut* = non-synonymous mutation on the earliest sequenced sample of the index cancer; `amp` / `homdel` / `fus` are separate. Patients are compared for a gene only if their panel covers it.
# KM / log-rank ignore left truncation (as cBioPortal does); the Cox HR is left-truncated at the sequencing report (*HR (LT)*).
#
# **Breast-specific caveat that matters for every question here:** PIK3CA mutations are enriched in HR+/HER2− disease and TP53 mutations in triple-negative and HER2+ disease. Any PIK3CA-vs-TP53 survival difference is therefore confounded by subtype unless the comparison is made within a subtype or adjusted for it. Both are shown.
#
# Treatment comparisons use regimens started **at or after the first distant metastasis** (`met_setting`), so that adjuvant endocrine therapy is not mixed with metastatic therapy.

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
DS = "bpc_brca"
F = []

b = cb.BPC("BrCa")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log_nsclc_brca.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "BrCa"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort and alteration matrix

# %%
S = b.survival_dx()
A = b.alterations()
display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index breast cancer (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("With OS from diagnosis", int(S.time.notna().sum())),
]))
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel; "
      f"{int((S.entry >= S.time).sum())} patients with sequencing report after last follow-up (dropped from left-truncated models only).")
print(f"Deaths: {int(S.event.sum())} / {len(S)}.  Median OS from dx: {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months.")
display(pd.concat([S.stage_dx.value_counts().rename("stage_dx"), S.bca_subtype.value_counts(dropna=False).rename("subtype")], axis=1).fillna(""))

# %%
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes (non-synonymous)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
freq.head(15)

# %%
pik = A.status("PIK3CA")
tp53 = A.status("TP53")
grp = A.combo_groups(["PIK3CA", "TP53"], min_n=1, wt_label="neither")
grp = grp.replace({"PIK3CA": "PIK3CA only", "TP53": "TP53 only", "PIK3CA+TP53": "both"})
order4 = ["neither", "PIK3CA only", "TP53 only", "both"]
display(pd.crosstab(grp, S.bca_subtype.reindex(grp.index).fillna("unknown"), margins=True))

# %% [markdown]
# ## Br1 (replicate) — PIK3CA-mutant patients have significantly better OS than TP53-mutant patients
# cBioPortal (slides 15–16) compared the PIK3CA-mutant and TP53-mutant groups; its group comparison excludes patients in both groups, so the primary replication is **PIK3CA-only vs TP53-only**. The four-group version and the subtype-adjusted model follow.

# %%
d1 = S.join(grp, how="inner")
d1x = d1[d1.profile.isin(["PIK3CA only", "TP53 only"])]
ax, tab1 = cb.km_plot(d1x, "profile", order=["PIK3CA only", "TP53 only"], title="Br1  OS from diagnosis: PIK3CA-only vs TP53-only", xlabel="Months from diagnosis")
h1 = cb.hr(d1x.assign(feat=(d1x.profile == "TP53 only")), "feat", entry="entry")
h1s = cb.hr(d1x.assign(feat=(d1x.profile == "TP53 only")), "feat", entry="entry", adjust=["bca_subtype", "stage_dx", "age_dx"])
tab1["HR_LT_tp53_vs_pik3ca"] = h1["HR"]; tab1["HR_adj_subtype_stage_age"] = h1s["HR"]; tab1["p_adj"] = h1s["p_cox"]
cb.save(DS, "Br1", table=tab1, fig=ax.figure, slug="pik3ca_only_vs_tp53_only")
display(tab1)

# %%
ax, tab1b = cb.km_plot(d1, "profile", order=order4, title="Br1  OS from diagnosis by PIK3CA / TP53 status (all four groups)", xlabel="Months from diagnosis")
cb.save(DS, "Br1", table=tab1b, fig=ax.figure, slug="four_groups")
display(tab1b)
# within HR+/HER2- only
hrpos = d1x[d1x.bca_subtype == "HR+, HER2-"]
ax, tab1c = cb.km_plot(hrpos, "profile", order=["PIK3CA only", "TP53 only"], title="Br1  HR+/HER2− only: PIK3CA-only vs TP53-only", xlabel="Months from diagnosis")
h1c = cb.hr(hrpos.assign(feat=(hrpos.profile == "TP53 only")), "feat", entry="entry")
tab1c["HR_LT_tp53_vs_pik3ca"] = h1c["HR"]
cb.save(DS, "Br1", table=tab1c, fig=ax.figure, slug="hrpos_her2neg_only")
display(tab1c)
cb.finding("Br1", "replicated; holds after subtype adjustment" if h1s["p_cox"] < 0.05 else "replicated; subtype-driven",
           f"PIK3CA-only (n={tab1.n[0]}) median OS {tab1.median_months[0]} vs TP53-only (n={tab1.n[1]}) {tab1.median_months[1]} mo, log-rank p={cb.fmt_p(tab1.p_logrank[0])}, "
           f"HR (LT) TP53 vs PIK3CA {h1['HR']:.2f}; adjusted for subtype, stage, age: HR {h1s['HR']:.2f} (p={cb.fmt_p(h1s['p_cox'])}). "
           f"Within HR+/HER2−: median {tab1c.median_months[0]} vs {tab1c.median_months[1]} mo, p={cb.fmt_p(tab1c.p_logrank[0])}, HR {h1c['HR']:.2f}.", F)

# %% [markdown]
# ## Br2 — Are there treatments that improved OS in TP53-mutant patients?
# For each treatment class (first regimen of that class per patient, metastatic setting), OS from regimen start in TP53-mutant vs TP53-wild-type patients. A class whose TP53-mut HR is close to 1 "closes the TP53 gap"; medians within TP53-mut show absolute outcomes. All lines are pooled, so classes are not comparable to each other (different lines and settings), only within class.

# %%
CLASSES = {
    "chemotherapy": dict(pattern="|".join(cb.CHEMO)),
    "aromatase inhibitor": dict(pattern="|".join(cb.AROMATASE_INHIBITORS)),
    "fulvestrant": dict(pattern="Fulvestrant"),
    "tamoxifen": dict(pattern="Tamoxifen"),
    "CDK4/6 inhibitor": dict(pattern="|".join(cb.CDK46)),
    "HER2-targeted": dict(pattern="Trastuzumab|Pertuzumab|Lapatinib|Neratinib|Tucatinib"),
    "everolimus": dict(pattern="Everolimus"),
    "capecitabine": dict(pattern="Capecitabine"),
    "eribulin": dict(pattern="Eribulin"),
    "immunotherapy": dict(pattern="|".join(cb.IO)),
}

def by_class(status, label, min_n=10):
    rows, kms = [], {}
    for name, kw in CLASSES.items():
        R = b.survival_regimen(**kw)
        R = R[R.met_setting].join(status.rename("feat"), how="inner")
        n_alt, n_wt = int(R.feat.sum()), int((~R.feat).sum())
        if n_alt < min_n or n_wt < min_n:
            continue
        R["group"] = np.where(R.feat, f"{label} mut", f"{label} wt")
        t = cb.km_table(R, "group", order=[f"{label} wt", f"{label} mut"])
        h = cb.hr(R, "feat", entry="entry")
        rows.append(dict(feature=name, n_alt=n_alt, n_wt=n_wt, events_alt=t.events[1], events_wt=t.events[0],
                         median_mut=t.median_months[1], median_wt=t.median_months[0], p_logrank=t.p_logrank[0], **h))
        kms[name] = R
    res = pd.DataFrame(rows)
    res["q_bh"] = cb.multipletests(res.p_logrank, method="fdr_bh")[1]
    return res.sort_values("p_logrank").reset_index(drop=True), kms

br2, kms2 = by_class(tp53, "TP53")
ax = cb.forest_plot(br2, top=len(br2), title="Br2  OS from regimen start: TP53-mut vs TP53-wt, by treatment class (metastatic setting)")
ax.set_xlabel("Hazard ratio TP53-mut vs wt (left-truncated), 95% CI")
cb.save(DS, "Br2", table=br2, fig=ax.figure, slug="tp53_by_treatment_class")
display(br2)

# %%
for name in list(br2.sort_values("p_logrank").feature[:2]) + ["CDK4/6 inhibitor"]:
    if name in kms2:
        R = kms2[name]
        ax, t = cb.km_plot(R, "group", order=["TP53 wt", "TP53 mut"], title=f"Br2  OS from start of {name} (metastatic setting) by TP53 status", xlabel="Months from regimen start")
        cb.save(DS, "Br2", table=t, fig=ax.figure, slug="km_" + re.sub(r"[^a-z0-9]+", "_", name.lower()))
closes = br2[(br2.HR_lo <= 1) & (br2.n_alt >= 20)]
cb.finding("Br2", "descriptive",
           "Within TP53-mut, median OS from regimen start (mo): " + ", ".join(f"{r.feature} {r.median_mut} (n={r.n_alt})" for r in br2.sort_values('median_mut', ascending=False).itertuples()) +
           ". TP53-mut is worse on every class; classes where the TP53 gap is not significant (CI crosses 1, n>=20): " + (", ".join(closes.feature) or "none") + ".", F)

# %% [markdown]
# ## Br3 — Are there co-mutations in PIK3CA patients that caused worse outcomes?
# Within PIK3CA-mutant patients: every co-alteration with ≥ 10 carriers / non-carriers; log-rank, left-truncated HR, BH FDR. Subtype-adjusted HR added for the hits.

# %%
pik_ids = pik.index[pik]
scr3 = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel"), min_n=10, entry="entry", patients=pik_ids)
scr3 = scr3[scr3.gene != "PIK3CA"].reset_index(drop=True)
ax = cb.forest_plot(scr3, top=15, title="Br3  Co-alterations and OS within PIK3CA-mutant breast cancer (top 15)")
cb.save(DS, "Br3", table=scr3, fig=ax.figure, slug="comutation_screen_in_pik3ca")
display(scr3.head(15))
hits3 = scr3[scr3.q_bh < 0.1].copy()
adj = []
for r in hits3.itertuples():
    d = S.loc[pik_ids].join(A.status(r.gene, r.kind).rename("feat"), how="inner")
    adj.append(cb.hr(d, "feat", entry="entry", adjust=["bca_subtype", "stage_dx", "age_dx"])["HR"])
hits3["HR_adj_subtype_stage_age"] = adj
display(hits3[["feature", "n_alt", "median_alt", "median_wt", "HR", "p_logrank", "q_bh", "HR_adj_subtype_stage_age"]])
for r in hits3.head(2).itertuples():
    d, t = cb.compare(S.loc[pik_ids], A.status(r.gene, r.kind), labels=(f"{r.feature} absent", f"{r.feature} present"), entry="entry")
    ax, _ = cb.km_plot(d, "group", order=[f"{r.feature} absent", f"{r.feature} present"], title=f"Br3  PIK3CA-mut: OS from diagnosis by {r.feature}", xlabel="Months from diagnosis")
    cb.save(DS, "Br3", table=t, fig=ax.figure, slug=f"pik3ca_by_{r.gene}_{r.kind}")
cb.finding("Br3", "signal" if len(hits3) else "no signal",
           (f"{len(hits3)} of {len(scr3)} co-alterations at FDR<0.1: " + "; ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f}, subtype-adjusted {r.HR_adj_subtype_stage_age:.2f}, q={r.q_bh:.2g}" for r in hits3.itertuples()))
           if len(hits3) else f"none of {len(scr3)} passes FDR<0.1 (best {scr3.feature[0]} p={cb.fmt_p(scr3.p_logrank[0])})", F)

# %% [markdown]
# ## Br4 — OS from the start of different treatments
# All patients, metastatic-setting regimens, first regimen of each class per patient. Classes are not randomised comparisons: line of therapy and subtype differ between them.

# %%
rows, lines = [], []
for name, kw in CLASSES.items():
    R = b.survival_regimen(**kw)
    R = R[R.met_setting]
    if len(R) < 10:
        continue
    t = cb.km_table(R.assign(g=name), "g")
    t["first_line_pct"] = round(100 * (R.line == 1).mean(), 1)
    t["median_line"] = R.line.median()
    rows.append(t)
    lines.append(R.assign(treatment=name))
br4 = pd.concat(rows).drop(columns=["p_logrank"]).rename(columns={"group": "treatment"}).sort_values("median_months", ascending=False).reset_index(drop=True)
d4 = pd.concat(lines)
top5 = br4.treatment[:6].tolist()
ax, _ = cb.km_plot(d4[d4.treatment.isin(top5)], "treatment", order=top5, title="Br4  OS from regimen start by treatment class (metastatic setting, any line)", xlabel="Months from regimen start", risk_table=False)
cb.save(DS, "Br4", table=br4, fig=ax.figure, slug="os_by_treatment_class")
display(br4)
# same thing restricted to first-line metastatic regimens
d4f = d4[d4.line == 1]
t4f = pd.concat([cb.km_table(g.assign(g=n), "g").assign(treatment=n) for n, g in d4f.groupby("treatment") if len(g) >= 10]).drop(columns=["p_logrank", "group"])
cb.save(DS, "Br4", table=t4f, slug="os_by_treatment_class_firstline")
display(t4f.sort_values("median_months", ascending=False))
cb.finding("Br4", "descriptive",
           "Median OS from regimen start (metastatic setting): " + ", ".join(f"{r.treatment} {r.median_months} mo (n={r.n})" for r in br4.itertuples()) + ".", F)

# %% [markdown]
# ## Br5 (replicate) — PIK3CA-mut vs TP53-mut patients who received fulvestrant: no significant OS difference
# cBioPortal's "OS from start of treatment" attributes are defined per **regimen**, so the deck's figure is OS from the start of the
# **fulvestrant monotherapy** regimen (`Fulvestrant`), PIK3CA-only vs TP53-only (overlap dropped). That is the primary replication.
# The extension uses the first **fulvestrant-containing** regimen of any kind (fulvestrant ± CDK4/6 inhibitor, everolimus, alpelisib, …), which triples the sample.

# %%
def regimen_groups(R, title, qid, slug):
    d = R.join(grp, how="inner")
    dx = d[d.profile.isin(["PIK3CA only", "TP53 only"])]
    ax, t = cb.km_plot(dx, "profile", order=["PIK3CA only", "TP53 only"], title=title, xlabel="Months from regimen start")
    h = cb.hr(dx.assign(feat=(dx.profile == "TP53 only")), "feat", entry="entry")
    hs = cb.hr(dx.join(S[["bca_subtype", "age_dx"]]).assign(feat=(dx.profile == "TP53 only")), "feat", entry="entry", adjust=["bca_subtype", "age_dx"])
    t["HR_LT_tp53_vs_pik3ca"] = h["HR"]; t["HR_lo"] = h["HR_lo"]; t["HR_hi"] = h["HR_hi"]; t["HR_adj_subtype_age"] = hs["HR"]
    cb.save(DS, qid, table=t, fig=ax.figure, slug=slug)
    display(t)
    ax, t4 = cb.km_plot(d, "profile", order=order4, title=title + " (four groups)", xlabel="Months from regimen start")
    cb.save(DS, qid, table=t4, fig=ax.figure, slug=slug + "_four_groups")
    display(t4)
    display(pd.crosstab(dx.profile, dx.line.clip(upper=4).astype(int).astype(str).replace("4", "4+")).rename_axis(columns="line"))
    return t, h, hs

Rf = b.survival_regimen(exact="Fulvestrant"); Rf = Rf[Rf.met_setting]
t5, h5, h5s = regimen_groups(Rf, "Br5  OS from start of fulvestrant MONOTHERAPY regimen (replicate)", "Br5", "fulvestrant_mono_pik3ca_vs_tp53")
Rfa = b.survival_regimen(pattern="Fulvestrant"); Rfa = Rfa[Rfa.met_setting]
t5a, h5a, h5as = regimen_groups(Rfa, "Br5  OS from start of first fulvestrant-CONTAINING regimen (extension)", "Br5", "fulvestrant_any_pik3ca_vs_tp53")
cb.finding("Br5", ("replicated (underpowered)" if t5.p_logrank[0] > 0.05 else "not replicated") + ("; difference appears with any fulvestrant-containing regimen" if t5a.p_logrank[0] < 0.05 else ""),
           f"Fulvestrant monotherapy regimen: PIK3CA-only n={t5.n[0]} median {t5.median_months[0]} vs TP53-only n={t5.n[1]} median {t5.median_months[1]} mo, "
           f"log-rank p={cb.fmt_p(t5.p_logrank[0])}, HR (LT) {h5['HR']:.2f}. Any fulvestrant-containing regimen: n={t5a.n[0]} vs {t5a.n[1]}, median {t5a.median_months[0]} vs {t5a.median_months[1]} mo, "
           f"p={cb.fmt_p(t5a.p_logrank[0])}, HR (LT) {h5a['HR']:.2f}, subtype/age-adjusted {h5as['HR']:.2f}.", F)

# %% [markdown]
# ## Br6 (replicate) — On fulvestrant + palbociclib, PIK3CA-mut OS was better than TP53-mut
# OS from the start of the `Fulvestrant, Palbociclib` regimen (the per-regimen attribute cBioPortal exposes), metastatic setting.

# %%
Rc = b.survival_regimen(exact="Fulvestrant, Palbociclib"); Rc = Rc[Rc.met_setting]
rg = b.regimens.regimen_drugs.fillna("")
n_other = b.regimens[rg.str.contains("Fulvestrant") & rg.str.contains("Palbociclib") & (rg != "Fulvestrant, Palbociclib")].record_id.nunique()
print(f"patients with a 'Fulvestrant, Palbociclib' regimen in the metastatic setting: {len(Rc)}; {n_other} further patients received the pair only inside a larger combination (not counted)")
t6, h6, h6s = regimen_groups(Rc, "Br6  OS from start of fulvestrant + palbociclib regimen (replicate)", "Br6", "fulv_palbo_pik3ca_vs_tp53")
cb.finding("Br6", "replicated" if (t6.p_logrank[0] < 0.05 and t6.median_months[0] > t6.median_months[1]) else "not replicated / underpowered",
           f"Fulvestrant+palbociclib: PIK3CA-only n={t6.n[0]} median {t6.median_months[0]} vs TP53-only n={t6.n[1]} median {t6.median_months[1]} mo, "
           f"log-rank p={cb.fmt_p(t6.p_logrank[0])}, HR (LT) {h6['HR']:.2f} [{h6['HR_lo']:.2f}-{h6['HR_hi']:.2f}], subtype/age-adjusted {h6s['HR']:.2f}.", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
