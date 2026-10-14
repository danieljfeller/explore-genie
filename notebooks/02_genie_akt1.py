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
# # AACR Project GENIE AKT1 Cohort — questions A1–A5
#
# Source: Jacob's *CBIO thoughts* deck, slides 9–11. Data: `brca_akt1_genie_2019` from genie.cbioportal.org (REST download).
# 428 breast cancer patients, one sample each, from 6 centres; curated treatment exposure flags (aromatase inhibitor, CDK4/6 inhibitor,
# fulvestrant, chemotherapy, …), duration of CDK4/6 and fulvestrant treatment with censoring status, and OS from initial diagnosis.
#
# **Design caveats that shape every answer here**
# - This is a **case-control cohort built around AKT1 mutations** (126 AKT1-mutant vs 302 wild-type, almost all ER+). Gene frequencies are not population frequencies and AKT1 effects are the cohort's design variable.
# - Treatment flags are "ever received" (any line, adjuvant or metastatic); there are no regimen start dates for aromatase inhibitors, so "OS from AI start" cannot be computed. The timeline gives the day of metastatic diagnosis and the day the sequenced specimen was taken.
# - Primary endpoint used: **OS from metastatic diagnosis**, left-truncated at specimen collection for the Cox HR. OS from initial diagnosis is shown where it replicates a cBioPortal figure.
# - 25% of samples come from panels whose gene lists are not in the GENIE BPC panel files (MDA, GRCC); those samples are assumed tested for every gene.

# %%
import sys, re, ast, warnings
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "cbio_lib.py").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cbio_lib as cb

cb.setup_plots()
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.3g}".format
pd.options.display.width = 200
DS = "genie_akt1"
F = []

P = cb.Portal("brca_akt1_genie_2019")
log = pd.read_csv(cb.DATA / "genie_portal" / "download_log.csv")
display(log[log.study_id == "brca_akt1_genie_2019"])

# %% [markdown]
# ## Cohort, endpoints and alteration matrix

# %%
S = P.survival()                       # time = OS months from initial diagnosis
tl = pd.read_csv(P.dir / "data_timeline.tsv", sep="\t")
tl["attr"] = tl.attributes.apply(lambda s: {d["key"]: d["value"] for d in ast.literal_eval(s)} if isinstance(s, str) else {})
met_day = tl[(tl.eventType == "STATUS") & (tl.attr.apply(lambda a: a.get("STATUS") == "Metastatic Diagnosis"))].groupby("patientId").startNumberOfDaysSinceDiagnosis.min()
spec_day = tl[tl.eventType == "SPECIMEN"].groupby("patientId").startNumberOfDaysSinceDiagnosis.min()
S["met_dx_mos"] = met_day.reindex(S.index) / 30.4375
S["specimen_mos"] = spec_day.reindex(S.index) / 30.4375
S["time_met"] = (S.time - S.met_dx_mos)
S["entry"] = S.specimen_mos                                  # left truncation, OS from initial dx
S["entry_met"] = (S.specimen_mos - S.met_dx_mos).clip(lower=0)  # left truncation, OS from metastatic dx
sm = P.samples
S["age_met"] = pd.to_numeric(sm.AGE_AT_MET_DIAGNOSIS.reindex(S.index), errors="coerce")
S["hr_status"] = S.HORMONE_RECEPTOR_STATUS
S["er_pos"] = S.hr_status.astype(str).str.startswith("ER+")
S["ai"] = S.AROMATASE_INHIBITOR_YN.eq("Yes")
S["cdk"] = S.CDK46_INHIBITOR_YN.eq("Yes")
S["fulv"] = S.FULVESTRANT_YN.eq("Yes")
S["akt1"] = sm.AKT_MUTATION_STATUS.reindex(S.index).eq("Mutant")
bad = S.time_met <= 0
print(f"patients {len(S)}; with metastatic-dx day {S.met_dx_mos.notna().sum()}; with specimen day {S.specimen_mos.notna().sum()}; "
      f"OS from met dx <= 0 (dropped from met-dx analyses): {int(bad.sum())}")
S.loc[bad, "time_met"] = np.nan
display(cb.consort([("Patients in study", len(S)), ("With OS from initial diagnosis", int(S.time.notna().sum())),
                    ("With OS from metastatic diagnosis", int(S.time_met.notna().sum())), ("Aromatase-inhibitor treated", int(S.ai.sum())),
                    ("AI + CDK4/6 inhibitor treated", int((S.ai & S.cdk).sum()))]))
display(pd.crosstab(S.hr_status.fillna("unknown"), S.ai, margins=True).rename_axis(columns="AI treated"))
display(pd.crosstab(S.CENTER, S.akt1, margins=True).rename_axis(columns="AKT1 mutant"))
A = P.alterations()
print(f"alteration matrix: {len(A.patients)} patients x {len(A.genes)} genes; samples with unknown panel: {A.n_unknown_panel}")
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes (AKT1 is enriched by design)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
print("median OS from met dx (all):", cb.km_table(S.assign(g="all"), "g", time="time_met").median_months[0], "months;",
      "from initial dx:", cb.km_table(S.assign(g="all"), "g").median_months[0], "months")

# %% [markdown]
# ## A2 (replicate) — Patients who received aromatase therapy lived longer than those who did not
# Slide 10 (cBioPortal: OS from initial diagnosis, AI-treated vs not). Then the same comparison from metastatic diagnosis, within ER+ patients, and left-truncated. "Ever received an AI" is a survivor-dependent exposure, so the naive curves overstate any benefit.

# %%
S["ai_grp"] = np.where(S.ai, "AI treated", "no AI")
order = ["no AI", "AI treated"]
ax, t2 = cb.km_plot(S, "ai_grp", order=order, title="A2  OS from INITIAL diagnosis by aromatase-inhibitor exposure (replicate)", xlabel="Months from initial diagnosis")
h2 = cb.hr(S, "ai"); t2["HR_naive"] = h2["HR"]
cb.save(DS, "A2", table=t2, fig=ax.figure, slug="replicate_from_initial_dx")
display(t2)
ax, t2b = cb.km_plot(S, "ai_grp", time="time_met", order=order, title="A2  OS from METASTATIC diagnosis by AI exposure", xlabel="Months from metastatic diagnosis")
cb.save(DS, "A2", table=t2b, fig=ax.figure, slug="from_met_dx")
display(t2b)
er = S[S.er_pos]
ax, t2c = cb.km_plot(er, "ai_grp", time="time_met", order=order, title="A2  ER+ only: OS from metastatic diagnosis by AI exposure", xlabel="Months from metastatic diagnosis")
cb.save(DS, "A2", table=t2c, fig=ax.figure, slug="er_pos_from_met_dx")
display(t2c)
models = pd.DataFrame([
    dict(model="naive, OS from initial dx, all", **cb.hr(S, "ai")),
    dict(model="OS from met dx, all", **cb.hr(S, "ai", time="time_met")),
    dict(model="OS from met dx, ER+ only", **cb.hr(er, "ai", time="time_met")),
    dict(model="OS from met dx, ER+, left-truncated at specimen, + age + HER2", **cb.hr(er, "ai", time="time_met", entry="entry_met", adjust=["age_met", "HER2_STATUSALL"])),
])
cb.save(DS, "A2", table=models, slug="cox_models")
display(models)
cb.finding("A2", "replicated, but exposure is survivor-dependent",
           f"From initial dx: AI-treated n={t2.n[1]} median {t2.median_months[1]} vs {t2.median_months[0]} mo, p={cb.fmt_p(t2.p_logrank[0])}, HR {h2['HR']:.2f}. "
           f"From metastatic dx in ER+ patients: median {t2c.median_months[1]} vs {t2c.median_months[0]} mo, p={cb.fmt_p(t2c.p_logrank[0])}; "
           f"left-truncated, age/HER2-adjusted HR {models.HR.iloc[-1]:.2f} (p={cb.fmt_p(models.p_cox.iloc[-1])}). The untreated group is small (n={t2.n[0]}) and enriched for ER− disease.", F)

# %% [markdown]
# ## A1 / A3 — In AI-treated patients, which mutations are associated with worse OS? PIK3CA first, then every gene
# Within AI-treated patients: OS from metastatic diagnosis, each gene feature (mutation / amplification / deep deletion) with ≥ 10 carriers and non-carriers; log-rank, left-truncated Cox HR, BH FDR. AKT1 is the cohort's design variable and is reported but should not be over-read.

# %%
ai = S[S.ai]
scr = cb.gene_screen(ai, A, kinds=("mut", "amp", "homdel"), min_n=10, time="time_met", event="event", entry="entry_met")
ax = cb.forest_plot(scr, top=15, title="A1  AI-treated: gene features vs OS from metastatic diagnosis (top 15)")
cb.save(DS, "A1", table=scr, fig=ax.figure, slug="gene_screen_in_ai_treated")
display(scr.head(15))
hits = scr[scr.q_bh < 0.1]
cb.finding("A1", "signal" if len(hits) else "no signal",
           (f"{len(hits)} of {len(scr)} features at FDR<0.1: " + "; ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()))
           if len(hits) else f"none of {len(scr)} features passes FDR<0.1 (best {scr.feature[0]} HR {scr.HR[0]:.2f} p={cb.fmt_p(scr.p_logrank[0])})", F)

# %%
rows = []
for g in ["PIK3CA", "AKT1", "TP53", "ESR1", "GATA3", "CDH1", "MAP3K1"]:
    d, t = cb.compare(ai, A.status(g), labels=(f"{g} wt", f"{g} mut"), time="time_met", event="event", entry="entry_met")
    rows.append(t.assign(gene=g))
    if g in ("PIK3CA", "AKT1", "TP53"):
        ax, _ = cb.km_plot(d, "group", time="time_met", order=[f"{g} wt", f"{g} mut"], title=f"A3  AI-treated: OS from metastatic diagnosis by {g} mutation", xlabel="Months from metastatic diagnosis")
        cb.save(DS, "A3", table=t, fig=ax.figure, slug=f"ai_by_{g}")
a3 = pd.concat(rows).set_index(["gene", "group"])
cb.save(DS, "A3", table=a3.reset_index(), slug="key_genes")
display(a3)
pk = a3.loc["PIK3CA"]
cb.finding("A3", "no difference" if pk.p_logrank.iloc[0] > 0.05 else "signal",
           f"AI-treated, PIK3CA-mut n={pk.n.iloc[1]} median OS from met dx {pk.median_months.iloc[1]} vs wt {pk.median_months.iloc[0]} mo, p={cb.fmt_p(pk.p_logrank.iloc[0])}, HR (LT) {pk.HR.iloc[1]:.2f} [{pk.HR_lo.iloc[1]:.2f}-{pk.HR_hi.iloc[1]:.2f}]. "
           + "; ".join(f"{g} HR {a3.loc[g].HR.iloc[1]:.2f} (p={cb.fmt_p(a3.loc[g].p_logrank.iloc[0])})" for g in ["AKT1", "TP53", "ESR1"]) + ".", F)

# %% [markdown]
# ## A4 — AI-treated patients with co-occurring PIK3CA + AKT1 + TP53 mutations
# Mutually exclusive profiles over the three genes (restricted to patients tested for all three), plus every pairwise combination.

# %%
prof = A.combo_groups(["PIK3CA", "AKT1", "TP53"], min_n=5, wt_label="none of the three")
d4 = ai.join(prof, how="inner")
display(d4.profile.value_counts().rename("AI-treated patients"))
order4 = [g for g in ["none of the three", "PIK3CA", "AKT1", "TP53", "PIK3CA+TP53", "AKT1+TP53", "PIK3CA+AKT1", "PIK3CA+AKT1+TP53", "other combination"] if g in set(d4.profile)]
ax, t4 = cb.km_plot(d4, "profile", time="time_met", order=order4, title="A4  AI-treated: OS from metastatic diagnosis by PIK3CA / AKT1 / TP53 profile", xlabel="Months from metastatic diagnosis", risk_table=False)
cb.save(DS, "A4", table=t4, fig=ax.figure, slug="profiles")
display(t4)
# pairwise: each combination vs the rest of AI-treated patients
rows = []
for combo in [("PIK3CA", "TP53"), ("AKT1", "TP53"), ("PIK3CA", "AKT1"), ("PIK3CA", "AKT1", "TP53")]:
    st = pd.concat([A.status(g).rename(g) for g in combo], axis=1, join="inner").all(axis=1)
    d, t = cb.compare(ai, st, labels=("rest", "+".join(combo)), time="time_met", event="event", entry="entry_met")
    rows.append(dict(combination="+".join(combo), n=t.n[1], events=t.events[1], median_with=t.median_months[1], median_rest=t.median_months[0],
                     p_logrank=t.p_logrank[0], HR=t.attrs["HR"], HR_lo=t.attrs["HR_lo"], HR_hi=t.attrs["HR_hi"]))
pair = pd.DataFrame(rows)
cb.save(DS, "A4", table=pair, slug="pairwise_combinations")
display(pair)
cb.finding("A4", "signal" if (pair.p_logrank < 0.05).any() else "no signal / underpowered",
           "; ".join(f"{r.combination} n={r.n} median {r.median_with} vs {r.median_rest} mo, HR {r.HR:.2f} (p={cb.fmt_p(r.p_logrank)})" for r in pair.itertuples()), F)

# %% [markdown]
# ## A5 — 126 patients received CDK4/6 inhibitors, 123 with an AI: survival of the combination, then which mutations predict poorer outcomes
# Two endpoints: OS from metastatic diagnosis, and **duration of CDK4/6 treatment** (`CDK_4_6_DOT`, event = discontinuation), which is the closest thing to a response measure in this dataset. Groups of ≥ 8 given n = 123. Reference: PMID 39754979.

# %%
ac = S[S.ai & S.cdk].copy()
ac["cdk_time"] = pd.to_numeric(ac.CDK_4_6_DOT, errors="coerce")
ac["cdk_event"] = ac.CDK_4_6_SENSORING_STATUS.eq("event").astype(int)
print(f"AI + CDK4/6: n={len(ac)}; AKT1-mutant {int(ac.akt1.sum())}; deaths {int(ac.event.sum())}; CDK4/6 discontinuations {int(ac.cdk_event.sum())}")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
cb.km_plot(ac.assign(g="AI + CDK4/6 inhibitor"), "g", time="time_met", ax=axes[0], title="A5  OS from metastatic diagnosis", xlabel="Months from metastatic diagnosis", risk_table=False)
cb.km_plot(ac.assign(g="AI + CDK4/6 inhibitor"), "g", time="cdk_time", event="cdk_event", ax=axes[1], title="A5  Time on CDK4/6 inhibitor (event = discontinuation)", xlabel="Months on CDK4/6 inhibitor", risk_table=False)
axes[1].set_ylabel("Still on CDK4/6 inhibitor")
fig.tight_layout()
t5 = pd.concat([cb.km_table(ac.assign(g="OS from met dx"), "g", time="time_met"), cb.km_table(ac.assign(g="time on CDK4/6i"), "g", time="cdk_time", event="cdk_event")])
cb.save(DS, "A5", table=t5, fig=fig, slug="combination_curves")
display(t5)

# %%
scr_os = cb.gene_screen(ac, A, kinds=("mut", "amp", "homdel"), min_n=8, time="time_met", event="event", entry="entry_met")
scr_dot = cb.gene_screen(ac, A, kinds=("mut", "amp", "homdel"), min_n=8, time="cdk_time", event="cdk_event")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
cb.forest_plot(scr_os, top=12, title="A5  AI+CDK4/6i: OS from metastatic dx", ax=axes[0])
cb.forest_plot(scr_dot, top=12, title="A5  AI+CDK4/6i: time on CDK4/6 inhibitor", ax=axes[1]); axes[1].set_xlabel("HR for discontinuation (altered vs not)")
fig.tight_layout()
cb.save(DS, "A5", table=scr_os, fig=fig, slug="gene_screen_os")
cb.save(DS, "A5", table=scr_dot, slug="gene_screen_time_on_cdk46")
display(scr_os.head(10)); display(scr_dot.head(10))
h5 = scr_os[scr_os.q_bh < 0.1]; h5d = scr_dot[scr_dot.q_bh < 0.1]
cb.finding("A5", "signal" if (len(h5) or len(h5d)) else "no signal at FDR<0.1",
           f"n={len(ac)}, median OS from met dx {t5.median_months.iloc[0]} mo, median time on CDK4/6i {t5.median_months.iloc[1]} mo. OS hits: " +
           (", ".join(f"{r.feature} HR {r.HR:.2f} q={r.q_bh:.2g}" for r in h5.itertuples()) or f"none (best {scr_os.feature[0]} HR {scr_os.HR[0]:.2f} p={cb.fmt_p(scr_os.p_logrank[0])})") +
           ". Time-on-treatment hits: " + (", ".join(f"{r.feature} HR {r.HR:.2f} q={r.q_bh:.2g}" for r in h5d.itertuples()) or f"none (best {scr_dot.feature[0]} HR {scr_dot.HR[0]:.2f} p={cb.fmt_p(scr_dot.p_logrank[0])})") + ".", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
