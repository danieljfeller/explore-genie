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
# # GENIE BPC CRC v2.0-public — questions C1–C4
#
# Source: Jacob's *CBIO thoughts* deck, slides 19–23. Data: AACR Project GENIE BPC CRC v2.0-public (Synapse).
#
# Conventions as in `README.md`: index date = diagnosis of the index colorectal cancer, OS in months; *X mut* = non-synonymous mutation;
# `amp` / `homdel` / `fus` separate; patients compared for a gene only if their panel covers it; KM / log-rank as cBioPortal, Cox HR left-truncated at the sequencing report (*HR (LT)*).
#
# **Sample timing.** The cancer-panel-test table gives, for every sequenced sample, the months from diagnosis to the pathology procedure that
# produced it (`dx_path_proc_cpt_mos`). A sample is *pre-treatment* for a regimen class if it was collected before the first regimen of that class
# started, *post-treatment* otherwise. Leucovorin marks the FOLFOX / FOLFIRI / CAPOX-type backbones.

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
DS = "bpc_crc"
F = []

b = cb.BPC("CRC")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "CRC"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort, alteration matrix and sample timing

# %%
S = b.survival_dx()
A = b.alterations()
display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index CRC (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("With OS from diagnosis", int(S.time.notna().sum())),
]))
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel; "
      f"{int((S.entry >= S.time).sum())} patients with sequencing report after last follow-up (dropped from left-truncated models only).")
print(f"Deaths: {int(S.event.sum())} / {len(S)}.  Median OS from dx: {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months.")
display(S.stage_dx.value_counts().rename("stage_dx"))
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes (non-synonymous)")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")

# %%
# first regimen start per class, per patient (months from diagnosis)
rg = b.regimens
def first_start(mask):
    return rg[mask].groupby("record_id").dx_reg_start_int_mos.min()
leu_start = first_start(rg.regimen_drugs.fillna("").str.contains("Leucovorin", case=False))
chemo_start = first_start(cb.contains_any(rg.regimen_drugs, cb.CHEMO))
samp = b.samples_all.copy()                                   # every sequenced sample of the index cancer
samp["collected_mos"] = samp.dx_path_proc_cpt_mos
samp["leu_start"] = samp.patient_id.map(leu_start)
samp["chemo_start"] = samp.patient_id.map(chemo_start)
samp["timing_leu"] = np.select([samp.leu_start.isna(), samp.collected_mos < samp.leu_start], ["no leucovorin regimen", "pre-leucovorin"], default="post-leucovorin")
samp["timing_chemo"] = np.select([samp.chemo_start.isna(), samp.collected_mos < samp.chemo_start], ["no chemotherapy", "pre-chemotherapy"], default="post-chemotherapy")
display(pd.crosstab(samp.timing_leu, samp.sample_type, margins=True))
print("regimens containing leucovorin:", rg.regimen_drugs.fillna("").str.contains("Leucovorin", case=False).sum(), "| top:")
display(rg[rg.regimen_drugs.fillna("").str.contains("Leucovorin", case=False)].regimen_drugs.value_counts().head(8))

# %% [markdown]
# ## C1 (replicate) — OS of patients biopsied before leucovorin vs after leucovorin was very different
# Slides 19–21. Patient-level: the earliest sample of the index cancer, classified pre / post the first leucovorin-containing regimen.
# A post-treatment biopsy can only happen in a patient who survived treatment, so OS from diagnosis is inflated for the post group
# (immortal time). Corrections: OS from sample collection, and a Cox model left-truncated at sample collection, stage- and age-adjusted.

# %%
first = samp.sort_values(["patient_id", "collected_mos"]).drop_duplicates("patient_id").set_index("patient_id")
S["timing_leu"] = first.timing_leu.reindex(S.index)
S["collected_mos"] = first.collected_mos.reindex(S.index)
S["time_from_sample"] = (S.time - S.collected_mos)
S["entry_sample"] = S.collected_mos.clip(lower=0)
order1 = ["pre-leucovorin", "post-leucovorin", "no leucovorin regimen"]
ax, t1 = cb.km_plot(S, "timing_leu", order=order1, title="C1  OS from diagnosis by biopsy timing relative to first leucovorin regimen (replicate)", xlabel="Months from diagnosis")
cb.save(DS, "C1", table=t1, fig=ax.figure, slug="os_from_dx_replicate")
display(t1)
two = S[S.timing_leu.isin(order1[:2])].copy()
two["post"] = two.timing_leu.eq("post-leucovorin")
ax, t1b = cb.km_plot(two, "timing_leu", time="time_from_sample", order=order1[:2], title="C1  OS from SAMPLE COLLECTION, pre vs post leucovorin", xlabel="Months from sample collection")
cb.save(DS, "C1", table=t1b, fig=ax.figure, slug="os_from_sample")
display(t1b)
models = pd.DataFrame([
    dict(model="naive, OS from dx", **cb.hr(two, "post")),
    dict(model="left-truncated at sample collection", **cb.hr(two, "post", entry="entry_sample")),
    dict(model="left-truncated + stage + age", **cb.hr(two, "post", entry="entry_sample", adjust=["stage_dx", "age_dx"])),
    dict(model="stage IV only, left-truncated + age", **cb.hr(two[two.stage_dx == "Stage IV"], "post", entry="entry_sample", adjust=["age_dx"])),
])
cb.save(DS, "C1", table=models, slug="cox_models")
display(models)
cb.finding("C1", "replicated, but it is immortal-time bias",
           f"From dx: pre n={t1.n[0]} median {t1.median_months[0]} vs post n={t1.n[1]} median {t1.median_months[1]} mo, p={cb.fmt_p(t1.p_logrank[0])}, naive HR post vs pre {models.HR[0]:.2f}. "
           f"From sample collection: median {t1b.median_months[0]} vs {t1b.median_months[1]} mo, p={cb.fmt_p(t1b.p_logrank[0])}; left-truncated + stage + age HR {models.HR[2]:.2f} (p={cb.fmt_p(models.p_cox[2])}).", F)

# %% [markdown]
# ## C2 — How do mutational profiles differ in samples taken before vs after chemotherapy?
# Sample-level comparison (every sequenced sample, classified against the first chemotherapy regimen of any kind and, separately, the first leucovorin regimen). Fisher test per gene feature, BH FDR. Samples from the same patient are not independent; C3 handles the paired subset.

# %%
As = b.alterations(rule="any")   # not used directly; build a sample-level matrix below
sample_level = samp[["sample_id"]].assign(patient_id=samp.sample_id)         # treat each sample as its own unit
AS = cb.build_alterations(sample_level, b.mut_long, b.cna_long, b.sv_long, b.panel_of, b.panel_genes)
res2 = {}
for col, post_label, pre_label in [("timing_chemo", "post-chemotherapy", "pre-chemotherapy"), ("timing_leu", "post-leucovorin", "pre-leucovorin")]:
    sub = samp[samp[col].isin([pre_label, post_label])].set_index("sample_id")
    anchor = sub[col].eq(post_label)
    co = AS.cooccurrence(anchor, kinds=("mut", "amp", "homdel"), min_n=5)
    co = co.rename(columns={"n_anchor": "n_post", "altered_in_anchor": "altered_post", "pct_in_anchor": "pct_post",
                            "n_rest": "n_pre", "altered_in_rest": "altered_pre", "pct_in_rest": "pct_pre"}).sort_values("p_fisher")
    res2[col] = co
    cb.save(DS, "C2", table=co, slug=f"{col}_frequencies")
    print(f"{col}: {int(anchor.sum())} post vs {int((~anchor).sum())} pre samples")
    display(co.head(12))
top = res2["timing_chemo"].head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 6))
y = np.arange(len(top))
ax.barh(y + 0.2, top.pct_post, height=0.4, color=cb.PALETTE[1], label="post-chemotherapy samples")
ax.barh(y - 0.2, top.pct_pre, height=0.4, color=cb.PALETTE[0], label="pre-chemotherapy samples")
ax.set_yticks(y); ax.set_yticklabels(top.gene + " " + top.kind); ax.set_xlabel("% of samples"); ax.legend()
ax.set_title("C2  Alteration frequency, pre- vs post-chemotherapy samples (top 15 by Fisher p)", loc="left", fontweight="bold")
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
cb.save(DS, "C2", fig=fig, slug="pre_vs_post_chemo")
sig2 = res2["timing_chemo"][res2["timing_chemo"].q_bh < 0.1]
cb.finding("C2", "signal" if len(sig2) else "no difference at FDR<0.1",
           "Pre vs post chemotherapy, FDR<0.1: " + (", ".join(f"{r.gene} {r.kind} {r.pct_pre}% -> {r.pct_post}%" for r in sig2.itertuples()) or "none") +
           f". Nominal top: {res2['timing_chemo'].gene.iloc[0]} {res2['timing_chemo'].kind.iloc[0]} {res2['timing_chemo'].pct_pre.iloc[0]}% -> {res2['timing_chemo'].pct_post.iloc[0]}% (p={cb.fmt_p(res2['timing_chemo'].p_fisher.iloc[0])}).", F)

# %% [markdown]
# ## C3 — How many patients have both pre- and post-chemotherapy biopsies, and how did their mutations change?
# Within-patient comparison: one pre- and one post-chemotherapy sample (earliest of each), restricted to genes on both samples' panels.

# %%
pre = samp[samp.timing_chemo == "pre-chemotherapy"].sort_values("collected_mos").drop_duplicates("patient_id").set_index("patient_id")
post = samp[samp.timing_chemo == "post-chemotherapy"].sort_values("collected_mos").drop_duplicates("patient_id").set_index("patient_id")
paired = pre.index.intersection(post.index)
print(f"patients with >=1 pre and >=1 post-chemotherapy sample: {len(paired)} (of {samp.patient_id.nunique()} sequenced patients)")
rows, per_patient = [], []
for pid in paired:
    s1, s2 = pre.loc[pid, "sample_id"], post.loc[pid, "sample_id"]
    for kind in ["mut", "amp", "homdel"]:
        M, T = AS.matrix(kind), AS.tested_for(kind)
        both_tested = T.loc[s1] & T.loc[s2]
        a1, a2 = M.loc[s1] & both_tested, M.loc[s2] & both_tested
        gained, lost, kept = a2 & ~a1, a1 & ~a2, a1 & a2
        per_patient.append(dict(patient_id=pid, kind=kind, months_between=round(post.loc[pid, "collected_mos"] - pre.loc[pid, "collected_mos"], 1),
                                n_tested=int(both_tested.sum()), kept=int(kept.sum()), gained=int(gained.sum()), lost=int(lost.sum()),
                                gained_genes=",".join(gained.index[gained]), lost_genes=",".join(lost.index[lost])))
        for g in gained.index[gained]: rows.append(dict(gene=g, kind=kind, change="gained"))
        for g in lost.index[lost]: rows.append(dict(gene=g, kind=kind, change="lost"))
pp = pd.DataFrame(per_patient)
ch = pd.DataFrame(rows)
ch = ch.groupby(["gene", "kind", "change"]).size().rename("n_patients").reset_index().sort_values("n_patients", ascending=False) if len(ch) else ch
cb.save(DS, "C3", table=pp, slug="per_patient_changes")
cb.save(DS, "C3", table=ch, slug="gene_changes")
display(pp.groupby("kind")[["kept", "gained", "lost"]].sum())
display(ch.head(20))
if len(pp):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    tot = pp.groupby("patient_id")[["gained", "lost"]].sum()
    ax.hist([tot.gained, tot.lost], bins=range(0, int(max(tot.max().max(), 1)) + 2), label=["gained after chemo", "lost after chemo"], color=cb.PALETTE[:2])
    ax.set_xlabel("alterations per patient"); ax.set_ylabel("patients"); ax.legend()
    ax.set_title("C3  Within-patient alteration changes, pre -> post chemotherapy", loc="left", fontweight="bold")
    cb.save(DS, "C3", fig=fig, slug="changes_hist")
cb.finding("C3", "descriptive",
           f"{len(paired)} patients have paired pre/post-chemotherapy samples. " +
           (f"Mutations gained {int(pp[pp.kind == 'mut'].gained.sum())}, lost {int(pp[pp.kind == 'mut'].lost.sum())}, kept {int(pp[pp.kind == 'mut'].kept.sum())}. Most often gained: " +
            ", ".join(f"{r.gene} {r.kind} (n={r.n_patients})" for r in ch[ch.change == 'gained'].head(5).itertuples()) if len(pp) else "") + ".", F)

# %% [markdown]
# ## C4 — How does mutation status affect survival: KRAS vs TP53 vs APC, plus co-mutations
# Single-gene comparisons (overlapping), then mutually exclusive profiles over the three genes (restricted to patients tested for all three), OS from diagnosis, HR (LT) adjusted for stage and age.

# %%
rows = []
for g in ["KRAS", "TP53", "APC", "NRAS", "BRAF", "PIK3CA", "SMAD4", "FBXW7"]:
    d, t = cb.compare(S, A.status(g), labels=(f"{g} wt", f"{g} mut"), entry="entry")
    hadj = cb.hr(d, "feat", entry="entry", adjust=["stage_dx", "age_dx"])
    t["HR_adj_stage_age"] = np.nan; t.loc[1, "HR_adj_stage_age"] = hadj["HR"]
    rows.append(t.assign(gene=g))
single = pd.concat(rows).set_index(["gene", "group"])
cb.save(DS, "C4", table=single.reset_index(), slug="single_genes")
display(single)
prof = A.combo_groups(["KRAS", "TP53", "APC"], min_n=10, wt_label="none of the three")
order4 = [g for g in ["none of the three", "KRAS", "TP53", "APC", "KRAS+TP53", "KRAS+APC", "TP53+APC", "KRAS+TP53+APC", "other combination"] if g in set(prof)]
d4 = S.join(prof, how="inner")
display(d4.profile.value_counts().rename("patients"))
ax, t4 = cb.km_plot(d4, "profile", order=order4, title="C4  OS from diagnosis by KRAS / TP53 / APC profile", xlabel="Months from diagnosis", risk_table=False)
d4c = d4.assign(profile=pd.Categorical(d4.profile, order4))
_, cox4 = cb.cox_fit(d4c, ["profile", "stage_dx", "age_dx"], entry="entry")
cb.save(DS, "C4", table=t4, fig=ax.figure, slug="profiles_os")
cb.save(DS, "C4", table=cox4, slug="profiles_cox_adjusted")
display(t4)
display(cox4.loc[cox4.index.str.startswith("profile")])
worst = cox4.loc[cox4.index.str.startswith("profile")].sort_values("HR", ascending=False).head(3)
cb.finding("C4", "signal" if (cox4.loc[cox4.index.str.startswith('profile')].p < 0.05).any() else "no signal",
           "Single genes (HR adj. stage/age): " + ", ".join(f"{g} {single.loc[(g, f'{g} mut'), 'HR_adj_stage_age']:.2f}" for g in ["KRAS", "TP53", "APC", "BRAF"]) +
           ". Profiles vs none, adjusted: " + ", ".join(f"{i.replace('profile_', '')} HR {r.HR:.2f} (p={cb.fmt_p(r.p)})" for i, r in worst.iterrows()) +
           f". Log-rank across profiles p={cb.fmt_p(t4.p_logrank[0])}.", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
