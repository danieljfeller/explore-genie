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
# # AACR Project GENIE v20.0-public — questions V1–V4 (ALK-altered patients)
#
# Source: Jacob's *CBIO thoughts* deck, slides 28–31. Data: GENIE v20.0-public main release (Synapse): clinical patient / sample tables,
# mutations, discrete CNA, structural variants, and the gene panel files. No treatment data.
#
# **What survival means here (V1).** GENIE releases, per patient, the interval from birth to death (`INT_DOD`) or to last contact
# (`INT_CONTACT`), and per sample the age at the sequencing report. The only index date available for every patient is therefore the
# **sequencing report**, and the derived endpoint is *OS from sequencing* = (INT_DOD or INT_CONTACT) − age at sequencing report.
# Diagnosis date cannot be inferred from age at sequencing. Intervals are censored at age > 89 or < 18 (kept at the boundary and flagged).
# Because every patient enters at sequencing, this endpoint is not left-truncated, but it is strongly affected by *when* in the disease
# course a centre sequences (at diagnosis vs at progression), which differs by centre and era.
#
# Conventions as in `README.md`. **ALK-altered** = any ALK alteration (structural variant / fusion, non-synonymous mutation, amplification, deep deletion),
# matching the cBioPortal query that produced the deck's 1,274 patients; fusion-only is the clinically meaningful subset and is shown separately.

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
DS = "genie_v20"
F = []

G = cb.GenieMain()
log = pd.read_csv(cb.DATA / "genie" / "download_log.csv")
display(log[log.cohort == "GENIE_main"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]])
print(f"patients {len(G.patients):,}, samples {len(G.samples_all):,}")

# %% [markdown]
# ## ALK-altered samples and patients

# %%
alk = G.gene_rows("ALK")
mut_s = set(alk["mut"][alk["mut"].Variant_Classification.isin(cb.NONSYN)].sample_id)
sv_s = set(alk["sv"].Sample_Id) if len(alk["sv"]) else set()
cand = G.samples_all[G.samples_all.sample_id.isin(mut_s | sv_s)]
# CNA for ALK across the release: the full matrix is too large to load, so stream the file and keep only the ALK row
alk_cna = None
with open(G._cna) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        if line.startswith("ALK\t"):
            alk_cna = pd.Series(line.rstrip("\n").split("\t")[1:], index=header[1:])
            break
alk_cna = pd.to_numeric(alk_cna, errors="coerce") if alk_cna is not None else pd.Series(dtype=float)
amp_s, del_s = set(alk_cna.index[alk_cna == 2]), set(alk_cna.index[alk_cna == -2])
alk_samples = G.samples_all[G.samples_all.sample_id.isin(mut_s | sv_s | amp_s | del_s)].copy()
alk_samples["alk_fusion"] = alk_samples.sample_id.isin(sv_s)
alk_samples["alk_mut"] = alk_samples.sample_id.isin(mut_s)
alk_samples["alk_amp"] = alk_samples.sample_id.isin(amp_s)
alk_samples["alk_homdel"] = alk_samples.sample_id.isin(del_s)
print(f"ALK-altered samples {len(alk_samples):,} from {alk_samples.patient_id.nunique():,} patients (deck: 1,153 samples / 1,274 patients)")
display(alk_samples[["alk_fusion", "alk_mut", "alk_amp", "alk_homdel"]].sum().rename("samples"))
display(alk_samples.CANCER_TYPE.value_counts().head(10).rename("samples by cancer type"))
display(pd.crosstab(alk_samples.CANCER_TYPE.where(alk_samples.CANCER_TYPE == "Non-Small Cell Lung Cancer", "other"), alk_samples.alk_fusion, margins=True).rename(columns={False: "no fusion", True: "fusion"}))

# %% [markdown]
# ## V1 — Can OS be derived at all?

# %%
S = G.survival_from_sequencing(alk_samples)
S["alk_fusion"] = S.sample_id.map(alk_samples.set_index("sample_id").alk_fusion)
S["nsclc"] = S.cancer_type.eq("Non-Small Cell Lung Cancer")
ok = S.time.notna() & S.event.notna()
display(cb.consort([("ALK-altered patients", len(S)), ("With vital status (DEAD true/false)", int(S.event.notna().sum())),
                    ("With usable interval (last contact / death after sequencing report)", int(ok.sum())),
                    ("… and not censored at an age boundary", int((ok & ~S.censored_interval).sum()))]))
display(pd.crosstab(S.center, ok, margins=True).rename(columns={False: "unusable", True: "usable"}))
S = S[ok]
print(f"ALK-altered, usable: n={len(S)}, deaths {int(S.event.sum())}, median derived OS from sequencing {cb.km_table(S.assign(g='all'), 'g').median_months[0]} months; "
      f"median age at sequencing {S.age_seq_years.median():.0f}")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
cb.km_plot(S, "nsclc", ax=axes[0], title="V1  Derived OS from sequencing: ALK-altered, NSCLC vs other", xlabel="Months from sequencing report", risk_table=False)
cb.km_plot(S[S.nsclc], "alk_fusion", ax=axes[1], title="V1  ALK-altered NSCLC: fusion vs other ALK alteration", xlabel="Months from sequencing report", risk_table=False)
fig.tight_layout()
cb.save(DS, "V1", table=cb.km_table(S, "nsclc"), fig=fig, slug="derived_os_overview")
# how far after diagnosis does sequencing happen? use BPC NSCLC (which has both dates) for ALK+ patients
try:
    bn = cb.BPC("NSCLC"); An = bn.alterations(); alk_bpc = An.status("ALK", "alt"); Sn = bn.survival_dx()
    lag = Sn.loc[alk_bpc.index[alk_bpc], "entry"]
    print(f"BPC NSCLC ALK-altered patients (n={len(lag)}): months from diagnosis to sequencing report median {lag.median():.1f}, IQR {lag.quantile(.25):.1f}-{lag.quantile(.75):.1f}")
except Exception as e:
    print("BPC NSCLC not available for the diagnosis-to-sequencing lag:", e)
cb.finding("V1", "feasible with caveats",
           f"OS from sequencing report is derivable for {len(S)} of {len(alk_samples.patient_id.unique())} ALK-altered patients (vital status + intervals released). "
           f"Diagnosis date cannot be inferred; in BPC NSCLC the diagnosis-to-sequencing lag is months to years and varies by stage. "
           f"Median derived OS: NSCLC {cb.km_table(S, 'nsclc').set_index('group').median_months.get(True)} mo.", F)

# %% [markdown]
# ## V2 — How does OS change with additional mutations in ALK-altered patients?
# ALK-altered NSCLC (one sample per patient, earliest), derived OS from sequencing; every co-alteration with ≥ 10 carriers / non-carriers;
# log-rank, Cox HR adjusted for age and sex, BH FDR.

# %%
nsclc = alk_samples[alk_samples.CANCER_TYPE == "Non-Small Cell Lung Cancer"].sort_values(["patient_id", "age_seq_days"]).drop_duplicates("patient_id")
A = G.alterations(nsclc[["sample_id", "patient_id"]])
mut_long, cna_long, sv_long = G.last_genomics
print(f"alteration matrix: {len(A.patients)} ALK-altered NSCLC patients x {len(A.genes)} genes; unknown panel {A.n_unknown_panel}")
Sn = S[S.index.isin(A.patients)].copy()
Sn["age"] = Sn.age_seq_years
scr = cb.gene_screen(Sn, A, kinds=("mut", "amp", "homdel", "fus"), min_n=10, adjust=["age", "sex"])
scr = scr[scr.gene != "ALK"].reset_index(drop=True)
ax = cb.forest_plot(scr, top=15, title="V2  ALK-altered NSCLC: co-alterations vs derived OS from sequencing (top 15)")
cb.save(DS, "V2", table=scr, fig=ax.figure, slug="comutation_screen")
display(scr.head(15))
fus_ids = Sn.index[Sn.alk_fusion]
scr_f = cb.gene_screen(Sn, A, kinds=("mut", "amp", "homdel"), min_n=10, adjust=["age", "sex"], patients=fus_ids)
scr_f = scr_f[scr_f.gene != "ALK"].reset_index(drop=True)
cb.save(DS, "V2", table=scr_f, slug="comutation_screen_fusion_only")
display(scr_f.head(10))
for g, k in [("TP53", "mut"), ("CDKN2A", "homdel")]:
    d, t = cb.compare(Sn, A.status(g, k), labels=(f"{g} {k} absent", f"{g} {k} present"), adjust=["age", "sex"])
    ax, _ = cb.km_plot(d, "group", order=[f"{g} {k} absent", f"{g} {k} present"], title=f"V2  ALK-altered NSCLC: derived OS by {g} {k}", xlabel="Months from sequencing report")
    cb.save(DS, "V2", table=t, fig=ax.figure, slug=f"by_{g}_{k}")
hits = scr[scr.q_bh < 0.1]
cb.finding("V2", "signal" if len(hits) else "no signal at FDR<0.1",
           f"ALK-altered NSCLC n={len(Sn)} ({len(fus_ids)} fusion). FDR<0.1: " +
           (", ".join(f"{r.feature} (n={r.n_alt}) HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()) or f"none (best {scr.feature[0]} HR {scr.HR[0]:.2f} p={cb.fmt_p(scr.p_logrank[0])})") + ".", F)

# %% [markdown]
# ## V3 — Full alteration profile of every ALK-altered patient, related to survival
# One row per patient with every mutation (protein change), structural variant and ±2 copy-number call; an oncoprint-style view of the
# 25 most frequent co-alterations in ALK-altered NSCLC, and the example profile from slide 31 (EML4-ALK + TP53 + STAT3 + EPHA3 + ERBB2 amp).

# %%
sid2pid = nsclc.set_index("sample_id").patient_id
prof_mut = mut_long.assign(patient_id=mut_long.sample_id.map(sid2pid)).dropna(subset=["patient_id"]).groupby("patient_id").apply(
    lambda d: "; ".join(sorted(f"{g} {p}" for g, p in zip(d.gene, d.HGVSp_Short.fillna("")))))
prof_sv = sv_long.assign(patient_id=sv_long.sample_id.map(sid2pid)).dropna(subset=["patient_id"]).groupby("patient_id").apply(
    lambda d: "; ".join(sorted(set(d.detail.fillna(d.gene).astype(str)))))
cna2 = cna_long[cna_long.value.abs() == 2].assign(patient_id=cna_long.sample_id.map(sid2pid)).dropna(subset=["patient_id"])
prof_cna = cna2.groupby("patient_id").apply(lambda d: "; ".join(sorted(f"{g} {'amp' if v == 2 else 'homdel'}" for g, v in zip(d.gene, d.value))))
profiles = pd.DataFrame({"mutations": prof_mut, "structural_variants": prof_sv, "copy_number": prof_cna}).reindex(A.patients).fillna("")
profiles = profiles.join(Sn[["time", "event", "cancer_type", "oncotree", "center", "age_seq_years"]], how="left")
cb.save(DS, "V3", table=profiles, slug="per_patient_profiles")
display(profiles.head(10))
example = A.status("ALK", "fus") & A.status("TP53") & (A.status("ERBB2", "amp"))
print(f"patients with EML4/ALK fusion + TP53 mut + ERBB2 amp: {int(example.sum())}; + STAT3 mut: {int((example & A.status('STAT3')).sum())}; + EPHA3 mut: {int((example & A.status('EPHA3')).sum())}")
# oncoprint-style matrix of the top co-alterations
fq = A.frequency(kinds=("mut", "amp", "homdel", "fus")).query("gene != 'ALK'").head(25)
feats = [(r.gene, r.kind) for r in fq.itertuples()]
M = pd.DataFrame({f"{g} {k}": A.matrix(k)[g] for g, k in feats}).astype(int)
M = M.loc[M.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(M.T.values, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#f0efec", cb.PALETTE[0]]), interpolation="nearest")
ax.set_yticks(range(len(M.columns))); ax.set_yticklabels([f"{c}  ({100 * M[c].mean():.0f}%)" for c in M.columns], fontsize=8)
ax.set_xticks([]); ax.set_xlabel(f"{len(M)} ALK-altered NSCLC patients"); ax.grid(False)
ax.set_title("V3  Co-alterations in ALK-altered NSCLC (blue = altered)", loc="left", fontweight="bold")
cb.save(DS, "V3", fig=fig, slug="oncoprint_top25")
# number of co-alterations vs OS
Sn["n_coalt"] = (A.matrix("alt").drop(columns="ALK", errors="ignore") & A.tested_for("alt").drop(columns="ALK", errors="ignore")).sum(axis=1).reindex(Sn.index)
Sn["coalt_group"] = pd.cut(Sn.n_coalt, [-1, 1, 3, 6, 1000], labels=["0-1", "2-3", "4-6", "7+"]).astype(str)
ax, t3 = cb.km_plot(Sn, "coalt_group", order=["0-1", "2-3", "4-6", "7+"], title="V3  ALK-altered NSCLC: derived OS by number of co-alterations", xlabel="Months from sequencing report")
cb.save(DS, "V3", table=t3, fig=ax.figure, slug="os_by_n_coalterations")
display(t3)
cb.finding("V3", "descriptive",
           f"Per-patient profiles written for {len(profiles)} ALK-altered NSCLC patients. Most frequent co-alterations: " +
           ", ".join(f"{r.gene} {r.kind} {r.pct}%" for r in fq.head(6).itertuples()) + f". OS by co-alteration burden: log-rank p={cb.fmt_p(t3.p_logrank[0])}.", F)

# %% [markdown]
# ## V4 — No treatment data in v20: can treatments be identified another way?
# Three indirect routes, each quantified below:
# 1. **Linkage to GENIE BPC.** BPC patients keep their GENIE IDs, so every ALK-altered BPC NSCLC patient has full regimen data (the Guardant-style ~300-patient study would be a comparable size).
# 2. **Resistance mutations as a footprint of prior ALK TKI** (G1202R, L1196M, I1171x, F1174x, G1269A, C1156Y, L1198F …): their presence implies the sample was taken after TKI exposure.
# 3. **Sample context**: sample type (primary vs metastasis), sequencing year (crizotinib approved 2011, alectinib 2015, lorlatinib 2018), and age.

# %%
try:
    bpc_alk = alk_bpc.index[alk_bpc]
    overlap = Sn.index.intersection(bpc_alk)
    print(f"ALK-altered NSCLC patients also in BPC NSCLC (full treatment history available): {len(overlap)} of {len(bpc_alk)} BPC ALK-altered patients")
    rg = bn.regimens[bn.regimens.record_id.isin(bpc_alk)]
    display(cb.contains_any(rg.regimen_drugs, cb.ALK_TKI).groupby(rg.record_id).any().value_counts().rename("BPC ALK-altered patients with an ALK TKI regimen"))
    display(rg[cb.contains_any(rg.regimen_drugs, cb.ALK_TKI)].regimen_drugs.value_counts().head(8))
except Exception as e:
    print("BPC NSCLC linkage unavailable:", e)
res_pat = r"G1202R|L1196M|I1171[A-Z]|F1174[A-Z]|G1269A|C1156Y|L1198F|D1203N|S1206[A-Z]|E1210K|F1245[A-Z]|G1128A|T1151|L1152[A-Z]"
alk_mut = alk["mut"][alk["mut"].sample_id.isin(nsclc.sample_id)]
resist = alk_mut[alk_mut.HGVSp_Short.fillna("").str.contains(res_pat, regex=True)]
print(f"ALK-altered NSCLC samples carrying a known ALK-TKI resistance mutation: {resist.sample_id.nunique()} of {len(nsclc)} ({100 * resist.sample_id.nunique() / len(nsclc):.1f}%)")
display(resist.HGVSp_Short.value_counts().head(10))
Sn["resistance_mut"] = Sn.sample_id.isin(resist.sample_id)
display(pd.crosstab(Sn.seq_year, Sn.resistance_mut, margins=True).rename(columns={False: "no", True: "resistance mutation"}))
d, t4 = cb.compare(Sn, Sn.resistance_mut, labels=("no resistance mutation", "ALK resistance mutation (post-TKI footprint)"), adjust=["age", "sex"])
ax, _ = cb.km_plot(d, "group", order=["no resistance mutation", "ALK resistance mutation (post-TKI footprint)"], title="V4  Derived OS by presence of an ALK-TKI resistance mutation", xlabel="Months from sequencing report")
cb.save(DS, "V4", table=t4, fig=ax.figure, slug="resistance_footprint")
display(t4)
cb.finding("V4", "partially feasible",
           f"Treatment cannot be recovered from v20 itself. Indirect handles: {resist.sample_id.nunique()} samples carry an ALK-TKI resistance mutation (post-TKI footprint; derived OS HR {t4.attrs['HR']:.2f}); "
           f"the BPC NSCLC release carries full regimen histories for its ALK-altered patients (see linkage count above); sample type and sequencing year give treatment-era context.", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
