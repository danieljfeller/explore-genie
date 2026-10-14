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
# # GENIE BPC NSCLC v2.0-public — questions N1–N6
#
# Source: Jacob's *CBIO thoughts* deck, slides 4–8. Data: AACR Project GENIE BPC NSCLC v2.0-public
# (Synapse analytic, clinical, timeline and genomic files; see `data/genie/download_log_nsclc_brca.csv`).
#
# **Conventions** (shared by every notebook, see `README.md`):
#
# - Index date = **diagnosis** of the index NSCLC. OS in months, event = death (`os_dx_status`). Treatment questions use OS from the start of the regimen (`tt_os_g`).
# - *X mut* = any non-synonymous somatic mutation in X on the patient's earliest sequenced sample of the index cancer. Amplification (+2), deep deletion (−2) and fusions are counted separately and labelled `amp`, `homdel`, `fus`.
# - A patient is compared for a gene only if that gene is on the panel of the sample used; untested patients are dropped rather than called wild-type.
# - Kaplan–Meier curves and log-rank p-values ignore the fact that patients enter GENIE only when sequenced (same as cBioPortal). The Cox hazard ratio reported next to each comparison is **left-truncated at the sequencing-report date** (labelled *HR (LT)*), which removes most of that immortal-time bias. Where the two disagree, trust the HR.
# - Gene screens use ≥ 10 patients per group and Benjamini–Hochberg FDR (`q_bh`). Treat q < 0.1 as "worth a look", not as a result.

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
DS = "bpc_nsclc"      # results/bpc_nsclc/
F = []                # one-line verdict per question, printed at the end

b = cb.BPC("NSCLC")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log_nsclc_brca.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "NSCLC"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort and alteration matrix

# %%
S = b.survival_dx()          # one row per patient: time (months from dx), event, entry (dx -> sequencing report), covariates
A = b.alterations()          # patient x gene matrices: A.mut / A.amp / A.homdel / A.fus (+ tested masks)

display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index NSCLC (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("With OS from diagnosis", int(S.time.notna().sum())),
]))
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel; "
      f"{int((S.entry >= S.time).sum())} patients have their sequencing report after the last follow-up date and are dropped from left-truncated models only.")
print(f"Deaths: {int(S.event.sum())} / {len(S)}.  Median OS from dx: {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months.")
display(pd.concat([S.stage_dx.value_counts().rename("stage_dx"), S.ca_hist_adeno_squamous.value_counts().rename("histology")], axis=1).fillna(""))

# %%
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes (non-synonymous)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
freq.head(15)

# %% [markdown]
# ## N1 — What is the OS of patients with EGFR mutations?
# EGFR-mutant vs EGFR-wild-type NSCLC, OS from diagnosis (slide 4).

# %%
egfr = A.status("EGFR")                      # True = non-synonymous EGFR mutation (patients tested for EGFR)
egfr_ids = egfr.index[egfr]
d, tab = cb.compare(S, egfr, labels=("EGFR wt", "EGFR mut"), entry="entry")
ax, _ = cb.km_plot(d, "group", order=["EGFR wt", "EGFR mut"], title="N1  OS from diagnosis by EGFR mutation status", xlabel="Months from diagnosis")
cb.save(DS, "N1", table=tab, fig=ax.figure, slug="egfr_vs_wt")
display(tab)

# %% [markdown]
# Sensitivity: stage IV at diagnosis only (EGFR status is mostly actionable in advanced disease), and by EGFR mutation subtype.

# %%
d4, tab4 = cb.compare(S[S.stage_dx == "Stage IV"], egfr, labels=("EGFR wt", "EGFR mut"), entry="entry")
display(tab4)

# EGFR subtype from the protein change of the sample used
def egfr_subtype(hgvsp):
    out = set()
    for h in hgvsp:
        h = str(h)
        pos = re.search(r"(\d+)", h)
        pos = int(pos.group(1)) if pos else -1
        if "L858R" in h: out.add("L858R")
        elif "T790M" in h: out.add("T790M")
        elif "del" in h and 729 <= pos <= 761: out.add("ex19del")
        elif ("ins" in h or "dup" in h) and 762 <= pos <= 823: out.add("ex20ins")
        else: out.add("other")
    if out & {"L858R", "ex19del"}: return "classic (L858R / ex19del)"
    if "ex20ins" in out: return "exon 20 insertion"
    return "atypical / other"

used = S.index.to_series().map(A.sample_of)
em = b.mut_long[(b.mut_long.gene == "EGFR") & b.mut_long.sample_id.isin(used)]
sub = em.groupby("sample_id").HGVSp_Short.apply(egfr_subtype)
S["egfr_subtype"] = used.map(sub)
ds = S.loc[egfr_ids].dropna(subset=["egfr_subtype"])
ax, tabs = cb.km_plot(ds, "egfr_subtype", title="N1  EGFR-mutant NSCLC: OS from diagnosis by EGFR mutation subtype", xlabel="Months from diagnosis",
                      order=["classic (L858R / ex19del)", "exon 20 insertion", "atypical / other"])
cb.save(DS, "N1", table=tabs, fig=ax.figure, slug="egfr_subtypes")
display(tabs)
cb.finding("N1", "confirmed",
           f"EGFR-mut (n={tab.n[1]}) median OS {tab.median_months[1]} vs {tab.median_months[0]} mo in EGFR-wt; log-rank p={cb.fmt_p(tab.p_logrank[0])}; "
           f"HR (LT) {tab.attrs['HR']:.2f} [{tab.attrs['HR_lo']:.2f}-{tab.attrs['HR_hi']:.2f}]. Stage IV only: HR {tab4.attrs['HR']:.2f}. "
           f"Classic sensitising mutations drive the benefit (median {tabs.set_index('group').median_months.get('classic (L858R / ex19del)')} mo).", F)

# %% [markdown]
# ## N2 — How is OS in EGFR+ patients affected by co-occurring mutations?
# Within EGFR-mutant patients, every co-alteration (mutation, amplification, deep deletion) with ≥ 10 carriers and ≥ 10 non-carriers is tested: log-rank, left-truncated Cox HR, BH FDR.

# %%
scr2 = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel"), min_n=10, entry="entry", patients=egfr_ids)
scr2 = scr2[scr2.gene != "EGFR"].reset_index(drop=True)
ax = cb.forest_plot(scr2, top=15, title="N2  Co-alterations and OS within EGFR-mutant NSCLC (top 15 by log-rank p)")
cb.save(DS, "N2", table=scr2, fig=ax.figure, slug="comutation_screen_in_egfr")
display(scr2.head(15))

# %%
hits = scr2[scr2.q_bh < 0.1]
for g, k in ([("TP53", "mut")] + [(r.gene, r.kind) for r in hits.itertuples() if r.gene != "TP53"][:2]):
    d, t = cb.compare(S.loc[egfr_ids], A.status(g, k), labels=(f"{g} {k} absent", f"{g} {k} present"), entry="entry")
    ax, _ = cb.km_plot(d, "group", order=[f"{g} {k} absent", f"{g} {k} present"], title=f"N2  EGFR-mut: OS from diagnosis by {g} {k}", xlabel="Months from diagnosis")
    cb.save(DS, "N2", table=t, fig=ax.figure, slug=f"egfr_by_{g}_{k}")
    display(t)
cb.finding("N2", "signal" if len(hits) else "no signal",
           (f"{len(hits)} of {len(scr2)} co-alterations at FDR<0.1: " + "; ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()))
           if len(hits) else f"none of {len(scr2)} co-alterations passes FDR<0.1 (best {scr2.feature[0]} p={cb.fmt_p(scr2.p_logrank[0])})", F)

# %% [markdown]
# ## N3 — Which alterations most commonly co-occur with EGFR, and do any convey a negative prognosis?
# Frequency of each co-alteration in EGFR-mutant vs EGFR-wild-type patients (Fisher test), joined to the prognostic screen from N2.

# %%
co = A.cooccurrence(egfr, kinds=("mut", "amp", "homdel"), min_n=10)
co = co[co.gene != "EGFR"].sort_values("altered_in_anchor", ascending=False)
top = co.head(20)
fig, ax = plt.subplots(figsize=(7.5, 7))
y = np.arange(len(top))[::-1]
ax.barh(y + 0.2, top.pct_in_anchor, height=0.4, color=cb.PALETTE[1], label="EGFR-mut")
ax.barh(y - 0.2, top.pct_in_rest, height=0.4, color=cb.PALETTE[0], label="EGFR-wt")
ax.set_yticks(y); ax.set_yticklabels(top.gene + " " + top.kind); ax.set_xlabel("% of tested patients"); ax.legend()
ax.set_title("N3  Most frequent co-alterations in EGFR-mutant NSCLC vs EGFR-wild-type", loc="left", fontweight="bold")
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
prog = scr2[["gene", "kind", "median_alt", "median_wt", "HR", "HR_lo", "HR_hi", "p_logrank", "q_bh"]].rename(
    columns={"median_alt": "median_os_with", "median_wt": "median_os_without", "HR": "HR_LT", "q_bh": "q_bh_os"})
n3 = top.rename(columns={"q_bh": "q_fisher", "n_anchor": "n_egfr_mut", "altered_in_anchor": "n_co_altered",
                         "pct_in_anchor": "pct_in_egfr_mut", "pct_in_rest": "pct_in_egfr_wt"}).merge(prog, on=["gene", "kind"], how="left")
cb.save(DS, "N3", table=n3, fig=fig, slug="cooccurrence_with_prognosis")
display(n3[["gene", "kind", "n_co_altered", "pct_in_egfr_mut", "pct_in_egfr_wt", "p_fisher", "median_os_with", "median_os_without", "HR_LT", "p_logrank", "q_bh_os"]])
bad = n3[(n3.q_bh_os < 0.1) & (n3.HR_LT > 1)]
cb.finding("N3", "signal" if len(bad) else "descriptive",
           "Top co-alterations in EGFR-mut: " + ", ".join(f"{r.gene} {r.kind} {r.pct_in_egfr_mut}%" for r in n3.head(5).itertuples()) +
           ("; negative prognosis at FDR<0.1: " + ", ".join(f"{r.gene} {r.kind} HR {r.HR_LT:.2f}" for r in bad.itertuples()) if len(bad) else "; none of the frequent co-alterations is prognostic at FDR<0.1"), F)

# %% [markdown]
# ## N4 (replicate) — In EGFR+ patients osimertinib clearly outperforms chemotherapy
# cBioPortal figure on slide 5, EGFR-mutant patients on osimertinib vs on chemotherapy. cBioPortal's "OS from start of treatment"
# attributes are defined **per regimen** (e.g. `Osimertinib`, `Carboplatin, Pemetrexed Disodium`), and its group comparison drops
# patients who fall in both groups. Three views are shown:
#
# 1. **OS from diagnosis**, patients grouped by the treatment they ever received (non-overlapping). This is the view that makes osimertinib look "clearly better", and it is immortal-time bias: osimertinib was mostly 2nd+ line here (T790M era), so its patients had already survived first-line therapy.
# 2. **OS from regimen start**, metastatic setting, any line: osimertinib monotherapy regimen vs first chemotherapy regimen without an EGFR TKI.
# 3. First-line regimens only.
#
# Plus OS from regimen start for each individual chemotherapy regimen (the way cBioPortal exposes it).

# %%
osi = b.survival_regimen(exact="Osimertinib")                      # the regimen cBioPortal calls "Osimertinib"
osi = osi[osi.index.isin(egfr_ids) & osi.met_setting]
chemo = b.survival_regimen(pattern="|".join(cb.CHEMO), exclude="|".join(cb.EGFR_TKI))
chemo = chemo[chemo.index.isin(egfr_ids) & chemo.met_setting]
both = osi.index.intersection(chemo.index)
print(f"EGFR-mut, metastatic setting: osimertinib monotherapy {len(osi)}; chemotherapy without EGFR TKI {len(chemo)}; received both {len(both)}")
# view 1: OS from diagnosis by treatment ever received (non-overlapping groups, as in cBioPortal's group comparison)
d40 = pd.concat([S.loc[osi.index.difference(both)].assign(group="received osimertinib (never chemo)"),
                 S.loc[chemo.index.difference(both)].assign(group="received chemotherapy (never osimertinib)"),
                 S.loc[both].assign(group="received both")])
ax, tab0 = cb.km_plot(d40, "group", order=["received osimertinib (never chemo)", "received both", "received chemotherapy (never osimertinib)"],
                      title="N4  EGFR-mut NSCLC: OS from DIAGNOSIS by treatment ever received (likely deck view)", xlabel="Months from diagnosis")
cb.save(DS, "N4", table=tab0, fig=ax.figure, slug="os_from_dx_by_treatment_received")
display(tab0)

# %%
# view 2: OS from regimen start, metastatic setting, any line
d4 = pd.concat([osi.assign(group="Osimertinib"), chemo.assign(group="Chemotherapy, no EGFR TKI")]).reset_index()
display(pd.crosstab(d4.group, d4.line.clip(upper=4).astype(int).astype(str).replace("4", "4+"), margins=True).rename_axis(columns="line of therapy"))
ax, tab = cb.km_plot(d4, "group", order=["Osimertinib", "Chemotherapy, no EGFR TKI"], title="N4  EGFR-mut NSCLC: OS from regimen start (metastatic setting, any line)", xlabel="Months from regimen start")
h = cb.hr(d4.assign(feat=(d4.group == "Osimertinib")), "feat", entry="entry")
tab["HR_LT_osi_vs_chemo"] = h["HR"]; tab["HR_lo"] = h["HR_lo"]; tab["HR_hi"] = h["HR_hi"]; tab["p_cox"] = h["p_cox"]
cb.save(DS, "N4", table=tab, fig=ax.figure, slug="osimertinib_vs_chemo_from_regimen_start")
display(tab)
# view 3: first-line only
d41 = d4[d4.line == 1]
tab1 = cb.km_table(d41, "group", order=["Osimertinib", "Chemotherapy, no EGFR TKI"])
h1 = cb.hr(d41.assign(feat=(d41.group == "Osimertinib")), "feat", entry="entry")
tab1["HR_LT_osi_vs_chemo"] = h1["HR"]
cb.save(DS, "N4", table=tab1, slug="osimertinib_vs_chemo_firstline")
display(tab1)
# per-regimen view (cBioPortal style): OS from start of each common regimen among EGFR-mut patients
rows = []
for reg in b.regimens[b.regimens.record_id.isin(egfr_ids)].regimen_drugs.value_counts().head(12).index:
    R = b.survival_regimen(exact=reg); R = R[R.index.isin(egfr_ids) & R.met_setting]
    if len(R) >= 10:
        rows.append(cb.km_table(R.assign(g=reg), "g").assign(first_line_pct=round(100 * (R.line == 1).mean(), 1)))
per_reg = pd.concat(rows).drop(columns="p_logrank").rename(columns={"group": "regimen"}).sort_values("median_months", ascending=False).reset_index(drop=True)
cb.save(DS, "N4", table=per_reg, slug="per_regimen_os_egfr_mut")
display(per_reg)
cb.finding("N4", "not supported as a treatment effect",
           f"From diagnosis (deck view): received-osimertinib median {tab0.median_months[0]} vs received-chemo {tab0.median_months[2]} mo, p={cb.fmt_p(tab0.p_logrank[0])} (immortal time: osimertinib is 2nd+ line in {100 - round(100 * (osi.line == 1).mean())}% of patients). "
           f"From regimen start, metastatic setting: osimertinib n={tab.n[0]} median {tab.median_months[0]} vs chemo n={tab.n[1]} median {tab.median_months[1]} mo, p={cb.fmt_p(tab.p_logrank[0])}, HR (LT) {h['HR']:.2f}. "
           f"First line only: n={tab1.n[0]} vs {tab1.n[1]}, median {tab1.median_months[0]} vs {tab1.median_months[1]} mo, p={cb.fmt_p(tab1.p_logrank[0])}.", F)

# %% [markdown]
# ## N5 — Mutation profiles of EGFR+ patients who respond well vs poorly to osimertinib
# Response is not curated directly. BPC progression-free survival counts any imaging or oncologist note of progression, so its median on
# osimertinib here is only ~5 months (far below trial PFS) and a PFS cut would label most patients "poor".
# **Default definition used (open question for Jacob):** time to next treatment for this cancer (`ttnt_ca_seq`; event = next regimen or death).
# *Good* = no next regimen or death for ≥ 12 months; *poor* = next regimen or death within 6 months; patients in between, or censored before 12 months, are excluded.
# All osimertinib-containing regimens count. The alteration profile comes from the patient's earliest sequenced sample, which may pre- or post-date osimertinib.

# %%
osi_all = b.survival_regimen(pattern="Osimertinib"); osi_all = osi_all[osi_all.index.isin(egfr_ids)]
good_lab, poor_lab = "good (>= 12 mo without next treatment)", "poor (next treatment or death < 6 mo)"
o = osi_all.dropna(subset=["ttnt_time", "ttnt_event"]).copy()
o["response"] = np.select([(o.ttnt_event == 1) & (o.ttnt_time < 6), o.ttnt_time >= 12], [poor_lab, good_lab], default="indeterminate")
display(o.response.value_counts().rename("EGFR-mut patients on osimertinib"))
gp = o[o.response != "indeterminate"]
ax, tab5 = cb.km_plot(gp.reset_index(), "response", order=[good_lab, poor_lab], title="N5  OS from osimertinib start by response group (sanity check)", xlabel="Months from osimertinib start")
anchor = gp.response.eq(good_lab)
prof5 = A.cooccurrence(anchor, kinds=("mut", "amp", "homdel"), min_n=3)
prof5 = prof5[prof5.gene != "EGFR"].rename(columns={"n_anchor": "n_good", "altered_in_anchor": "altered_good", "pct_in_anchor": "pct_good",
                                                    "n_rest": "n_poor", "altered_in_rest": "altered_poor", "pct_in_rest": "pct_poor"}).sort_values("p_fisher")
cb.save(DS, "N5", table=prof5, fig=ax.figure, slug="good_vs_poor_responder_profiles")
display(prof5.head(15))

# %%
# Continuous version: which baseline co-alterations shorten time to next treatment, and PFS-I, on osimertinib (all EGFR-mut osimertinib-treated patients)
scr5 = cb.gene_screen(osi_all, A, kinds=("mut", "amp", "homdel"), min_n=8, time="ttnt_time", event="ttnt_event", entry="entry", patients=osi_all.index)
scr5 = scr5[scr5.gene != "EGFR"].reset_index(drop=True)
scr5p = cb.gene_screen(osi_all, A, kinds=("mut", "amp", "homdel"), min_n=8, time="pfs_i_time", event="pfs_i_event", entry="entry", patients=osi_all.index)
scr5p = scr5p[scr5p.gene != "EGFR"].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
cb.forest_plot(scr5, top=12, title="N5  Time to next treatment on osimertinib, by baseline co-alteration", ax=axes[0])
axes[0].set_xlabel("HR for next treatment or death (altered vs not), 95% CI")
cb.forest_plot(scr5p, top=12, title="N5  PFS-I on osimertinib, by baseline co-alteration", ax=axes[1])
axes[1].set_xlabel("HR for imaging progression or death (altered vs not), 95% CI")
fig.tight_layout()
cb.save(DS, "N5", table=scr5, fig=fig, slug="ttnt_screen_on_osimertinib")
cb.save(DS, "N5", table=scr5p, slug="pfs_i_screen_on_osimertinib")
display(scr5.head(12)); display(scr5p.head(8))
sig5 = prof5[prof5.q_bh < 0.1]; sig5b = scr5[scr5.q_bh < 0.1]; sig5c = scr5p[scr5p.q_bh < 0.1]
cb.finding("N5", "signal" if (len(sig5) or len(sig5b) or len(sig5c)) else "no robust signal",
           f"good n={int(anchor.sum())}, poor n={int((~anchor).sum())}. Profile differences at FDR<0.1: " +
           (", ".join(f"{r.gene} {r.kind} {r.pct_good}% vs {r.pct_poor}%" for r in sig5.itertuples()) or "none") +
           ". Time-to-next-treatment screen at FDR<0.1: " + (", ".join(f"{r.feature} HR {r.HR:.2f}" for r in sig5b.itertuples()) or "none") +
           ". PFS-I screen at FDR<0.1: " + (", ".join(f"{r.feature} HR {r.HR:.2f}" for r in sig5c.itertuples()) or "none") +
           f". Nominal top (TTNT): {scr5.feature[0]} HR {scr5.HR[0]:.2f} p={cb.fmt_p(scr5.p_logrank[0])}.", F)

# %% [markdown]
# ## N6 — EGFR vs KEAP1 vs STK11 vs KEAP1+STK11: OS and response to different treatments
# Mutually exclusive profiles over the three genes, restricted to patients tested for all three (slides 6–8). cBioPortal dropped the KEAP1+STK11 overlap; here it is its own group. Patients with other combinations (e.g. EGFR+KEAP1) are listed but not plotted.

# %%
prof6 = A.combo_groups(["EGFR", "KEAP1", "STK11"], min_n=10, wt_label="none of the three")
order6 = ["none of the three", "EGFR", "KEAP1", "STK11", "KEAP1+STK11"]
display(prof6.value_counts().rename("patients"))
d6 = S.join(prof6, how="inner")
d6 = d6[d6.profile.isin(order6)]
ax, tab6 = cb.km_plot(d6, "profile", order=order6, title="N6  OS from diagnosis by mutation profile", xlabel="Months from diagnosis")
d6c = d6.assign(profile=pd.Categorical(d6.profile, order6))
_, cox6 = cb.cox_fit(d6c, ["profile"], entry="entry")
_, cox6a = cb.cox_fit(d6c, ["profile", "stage_dx", "age_dx"], entry="entry")
cox6["model"] = "left-truncated"; cox6a["model"] = "left-truncated + stage + age"
cb.save(DS, "N6", table=tab6, fig=ax.figure, slug="profiles_os")
cb.save(DS, "N6", table=pd.concat([cox6, cox6a]), slug="profiles_cox")
display(tab6)
display(pd.concat([cox6, cox6a]).loc[lambda t: t.index.str.startswith("profile")])

# %%
classes = {
    "immunotherapy-containing": dict(pattern="|".join(cb.IO)),
    "platinum chemotherapy, no IO or TKI": dict(pattern="Carboplatin|Cisplatin", exclude="|".join(cb.IO + cb.EGFR_TKI)),
    "EGFR TKI": dict(pattern="|".join(cb.EGFR_TKI)),
}
rows = []
for name, kw in classes.items():
    R = b.survival_regimen(**kw).join(prof6, how="inner")
    R = R[R.profile.isin(order6)]
    ax, t = cb.km_plot(R, "profile", order=order6, title=f"N6  OS from start of {name} regimen, by profile", xlabel="Months from regimen start")
    t["treatment"] = name
    rows.append(t)
    cb.save(DS, "N6", table=t, fig=ax.figure, slug="by_" + re.sub(r"[^a-z]+", "_", name))
t6 = pd.concat(rows)
piv = t6.pivot(index="group", columns="treatment", values=["n", "median_months"]).reindex(order6)
display(piv)
io = t6[t6.treatment == "immunotherapy-containing"].set_index("group")
cb.finding("N6", "signal",
           f"OS from dx: median none={tab6.median_months[0]}, EGFR={tab6.median_months[1]}, KEAP1={tab6.median_months[2]}, STK11={tab6.median_months[3]}, KEAP1+STK11={tab6.median_months[4]} mo; "
           f"log-rank p={cb.fmt_p(tab6.p_logrank[0])}. Adjusted HR vs none: " +
           ", ".join(f"{i.replace('profile_', '')} {r.HR:.2f}" for i, r in cox6a.iterrows() if i.startswith("profile")) +
           f". On immunotherapy: median none={io.median_months.get('none of the three')}, KEAP1={io.median_months.get('KEAP1')}, STK11={io.median_months.get('STK11')}, "
           f"KEAP1+STK11={io.median_months.get('KEAP1+STK11')} mo (p={cb.fmt_p(io.p_logrank.iloc[0])}).", F)

# %% [markdown]
# ## Summary of findings
# One line per question. "signal" means something passed the pre-specified threshold; "descriptive" means the question asked for a ranking, not a test.

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
