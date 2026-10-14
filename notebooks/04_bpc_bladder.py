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
# # GENIE BPC BLADDER v1.0-public — questions B1–B2
#
# Source: Jacob's *CBIO thoughts* deck, slide 14. Data: AACR Project GENIE BPC BLADDER v1.0-public (Synapse).
#
# Conventions as in `README.md`: index date = diagnosis of the index bladder cancer, OS in months; treatment questions use OS from regimen start in
# the metastatic setting. **CDKN2A/B altered** = deep deletion or non-synonymous mutation of CDKN2A or CDKN2B (deep deletion is the usual mechanism).
# Patients compared for a gene only if their panel covers it; KM / log-rank as cBioPortal, Cox HR left-truncated at the sequencing report (*HR (LT)*).

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
DS = "bpc_bladder"
F = []

b = cb.BPC("BLADDER")
pd.Series(b.summary())

# %% [markdown]
# ## Data pin

# %%
log = pd.read_csv(cb.DATA / "genie" / "download_log.csv")
print("release_version:", b.index.release_version.unique())
log[log.cohort == "BLADDER"][[c for c in ["category", "filename", "synapse_id", "version", "md5"] if c in log.columns]]

# %% [markdown]
# ## Cohort and alteration matrix
# The deck asks about *bladder urothelial carcinoma*; the histology / OncoTree code of the sequenced sample is used to restrict to urothelial tumours.

# %%
S = b.survival_dx()
A = b.alterations()
onc = S.index.to_series().map(b.samples.cpt_oncotree_code)
S["oncotree"] = onc
display(S.oncotree.value_counts().head(10).rename("OncoTree code of sample used"))
uro_codes = {"BLCA", "UTUC", "UCU", "BLSC", "URCC", "UCA", "UPA", "BLAD"}      # urothelial carcinoma OncoTree codes (bladder and upper tract)
hist_code = pd.to_numeric(S.get("naaccr_histology_cd", pd.Series(index=S.index, dtype=float)), errors="coerce")
hist_text = S.get("ca_histology", pd.Series(index=S.index, dtype=object)).astype(str)
S["urothelial"] = S.oncotree.isin(uro_codes) | hist_code.between(8120, 8131) | hist_text.str.contains("rothelial|ransitional", na=False)   # ICD-O-3 8120–8131 = urothelial / transitional cell
if S.urothelial.sum() == 0:
    raise RuntimeError(f"no urothelial patients found; OncoTree codes present: {S.oncotree.value_counts().head(10).to_dict()}")
display(cb.consort([
    ("Patients in release", len(b.patients)),
    ("Index bladder cancer (first index cancer per patient)", len(b.index)),
    ("With a sequenced sample of the index cancer", int(S.entry.notna().sum())),
    ("Urothelial carcinoma (OncoTree / histology)", int(S.urothelial.sum())),
]))
U = S[S.urothelial].copy()
print(f"{b.n_multi_index} patients had two index cancers (first used); {A.n_unknown_panel} samples with unknown panel. Deaths: {int(U.event.sum())} / {len(U)}; "
      f"median OS from dx: {cb.km_table(U.assign(g='all'), 'g').median_months[0]} months.")
display(U.stage_dx.value_counts().rename("stage_dx"))
freq = A.frequency(kinds=("mut", "amp", "homdel", "fus"), patients=U.index)
ax = cb.bar_freq(freq, top=25, title="Most frequent alterations, urothelial carcinoma")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")
freq.head(15)

# %% [markdown]
# ## B1 — How do bladder urothelial carcinoma patients do with CDKN2A or CDKN2B alterations?

# %%
cdk = (A.status("CDKN2A", "homdel") | A.status("CDKN2A", "mut") | A.status("CDKN2B", "homdel") | A.status("CDKN2B", "mut")).rename("cdkn2ab")
d, t1 = cb.compare(U, cdk, labels=("CDKN2A/B intact", "CDKN2A/B altered"), entry="entry", adjust=["stage_dx", "age_dx"])
ax, _ = cb.km_plot(d, "group", order=["CDKN2A/B intact", "CDKN2A/B altered"], title="B1  Urothelial carcinoma: OS from diagnosis by CDKN2A/B alteration", xlabel="Months from diagnosis")
cb.save(DS, "B1", table=t1, fig=ax.figure, slug="cdkn2ab_vs_intact")
display(t1)
rows = []
for g, k in [("CDKN2A", "homdel"), ("CDKN2B", "homdel"), ("CDKN2A", "mut"), ("MTAP", "homdel")]:
    _, t = cb.compare(U, A.status(g, k), labels=("absent", "present"), entry="entry")
    rows.append(t.assign(feature=f"{g} {k}"))
parts = pd.concat(rows).set_index(["feature", "group"])
cb.save(DS, "B1", table=parts.reset_index(), slug="components")
display(parts)
# stage IV only
d4, t1b = cb.compare(U[U.stage_dx == "Stage IV"], cdk, labels=("CDKN2A/B intact", "CDKN2A/B altered"), entry="entry", adjust=["age_dx"])
display(t1b)
cb.finding("B1", "signal" if t1.attrs["p_cox"] < 0.05 else "no signal",
           f"Urothelial n={len(d)}: CDKN2A/B altered n={t1.n[1]} median OS {t1.median_months[1]} vs {t1.median_months[0]} mo, log-rank p={cb.fmt_p(t1.p_logrank[0])}, "
           f"HR (LT, stage/age-adjusted) {t1.attrs['HR']:.2f} [{t1.attrs['HR_lo']:.2f}-{t1.attrs['HR_hi']:.2f}]; stage IV only HR {t1b.attrs['HR']:.2f} (p={cb.fmt_p(t1b.attrs['p_cox'])}).", F)

# %% [markdown]
# ## B2 — How is OS affected by different treatments, especially in the CDKN2A/B group?
# First regimen of each class per patient (metastatic setting), OS from regimen start; within each class, CDKN2A/B altered vs intact.

# %%
CLASSES = {
    "platinum chemotherapy": dict(pattern="Cisplatin|Carboplatin"),
    "cisplatin-based": dict(pattern="Cisplatin"),
    "carboplatin-based": dict(pattern="Carboplatin", exclude="Cisplatin"),
    "immunotherapy": dict(pattern="|".join(cb.IO)),
    "enfortumab vedotin": dict(pattern="Enfortumab"),
    "FGFR inhibitor": dict(pattern="Erdafitinib|Infigratinib|Pemigatinib"),
    "taxane": dict(pattern="Paclitaxel|Docetaxel|Nabpaclitaxel"),
    "gemcitabine-containing": dict(pattern="Gemcitabine"),
}
rows, kms = [], {}
for name, kw in CLASSES.items():
    R = b.survival_regimen(**kw)
    R = R[R.met_setting & R.index.isin(U.index)].join(cdk.rename("feat"), how="inner")
    if len(R) < 10:
        continue
    t = cb.km_table(R.assign(g=name), "g")
    row = dict(treatment=name, n=t.n[0], events=t.events[0], median_months=t.median_months[0], first_line_pct=round(100 * (R.line == 1).mean(), 1))
    n_alt, n_wt = int(R.feat.sum()), int((~R.feat).sum())
    if n_alt >= 10 and n_wt >= 10:
        R["group"] = np.where(R.feat, "CDKN2A/B altered", "CDKN2A/B intact")
        tt = cb.km_table(R, "group", order=["CDKN2A/B intact", "CDKN2A/B altered"])
        h = cb.hr(R, "feat", entry="entry")
        row.update(n_altered=n_alt, median_altered=tt.median_months[1], median_intact=tt.median_months[0], p_logrank=tt.p_logrank[0], HR_LT=h["HR"], HR_lo=h["HR_lo"], HR_hi=h["HR_hi"])
        kms[name] = R
    rows.append(row)
b2 = pd.DataFrame(rows)
cb.save(DS, "B2", table=b2, slug="os_by_treatment_class_and_cdkn2ab")
display(b2)
for name in [n for n in ["platinum chemotherapy", "immunotherapy", "enfortumab vedotin"] if n in kms]:
    ax, t = cb.km_plot(kms[name], "group", order=["CDKN2A/B intact", "CDKN2A/B altered"], title=f"B2  OS from start of {name} by CDKN2A/B status", xlabel="Months from regimen start")
    cb.save(DS, "B2", table=t, fig=ax.figure, slug="km_" + re.sub(r"[^a-z]+", "_", name))
sig = b2.dropna(subset=["p_logrank"]) if "p_logrank" in b2 else b2.iloc[0:0]
sig = sig[sig.p_logrank < 0.05]
cb.finding("B2", "descriptive",
           "Median OS from regimen start (urothelial, metastatic setting): " + ", ".join(f"{r.treatment} {r.median_months} mo (n={r.n})" for r in b2.itertuples()) +
           ". CDKN2A/B altered vs intact within class: " + (", ".join(f"{r.treatment} HR {r.HR_LT:.2f} (p={cb.fmt_p(r.p_logrank)})" for r in b2.dropna(subset=['HR_LT']).itertuples()) or "no class with >=10 per group") + ".", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
