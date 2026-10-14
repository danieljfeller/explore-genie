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
# # Metastatic Breast Cancer 2013–2016 (DFCI, Garrido-Castro et al., CCR 2020) — exploratory, no deck questions
#
# This study is on slide 3 of Jacob's deck but has no questions attached. The cells below profile the dataset and run the
# breast-cancer questions from the BPC BrCa notebook (Br1, Br3) on it as an **independent replication cohort**, so Jacob can decide
# whether it is worth keeping.
#
# Data: `mbc_genie_2020` from genie.cbioportal.org. 856 patients, one sample each (OncoPanel v1/v2), **OS from metastatic diagnosis**
# (`OS_MONTHS`, the attribute the portal's KM plot uses) and from initial diagnosis, subtype at initial and metastatic diagnosis, sample
# type and site, timing of the biopsy relative to first-line systemic therapy, and adjuvant-therapy flags. No regimen table and no
# sequencing date, so no left truncation is possible.

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
DS = "dfci_mbc"
F = []

P = cb.Portal("mbc_genie_2020")
log = pd.read_csv(cb.DATA / "genie_portal" / "download_log.csv")
display(log[log.study_id == "mbc_genie_2020"])

# %% [markdown]
# ## Cohort

# %%
S = P.survival()                                  # time = OS months from metastatic diagnosis
S["time_init"] = pd.to_numeric(S.OS_MONTHS_INIT_DIAGNOSIS, errors="coerce")
S["subtype"] = S.SUBTYPE_METS.fillna("Unknown")
S["age_met"] = pd.to_numeric(S.AGE, errors="coerce")
S["dfi"] = pd.to_numeric(S.DFI_MONTHS, errors="coerce")
sm = P.samples
S["sample_type"] = sm.SAMPLE_TYPE.reindex(S.index)
S["biopsy_timing"] = S["1ST_LINE_TO_BIOPSY_<=_30"].fillna("not stated")
A = P.alterations()
print(f"patients {len(S)}, deaths {int(S.event.sum())}, median OS from met dx {cb.km_table(S.assign(g='all'), 'g').median_months[0]} mo; "
      f"alteration matrix {len(A.patients)} x {len(A.genes)}; unknown panel {A.n_unknown_panel}")
display(pd.crosstab(S.subtype, S.sample_type, margins=True))
display(S.biopsy_timing.value_counts().rename("biopsy timing vs first-line therapy"))
freq = A.frequency(kinds=("mut", "amp", "homdel"))
ax = cb.bar_freq(freq[freq.kind == "mut"], top=20, title="Most frequently mutated genes")
cb.save(DS, "cohort", table=freq, fig=ax.figure, slug="gene_frequency")

# %% [markdown]
# ## OS from metastatic diagnosis by subtype (context for everything below)

# %%
order = ["HER2-/HR+", "HER2+/HR+", "HER2+/HR-", "TNBC", "Unknown"]
ax, t0 = cb.km_plot(S, "subtype", order=order, title="MBC  OS from metastatic diagnosis by subtype", xlabel="Months from metastatic diagnosis", risk_table=False)
cb.save(DS, "M0", table=t0, fig=ax.figure, slug="os_by_subtype")
display(t0)

# %% [markdown]
# ## M1 — Br1 replicated here: PIK3CA-only vs TP53-only, OS from metastatic diagnosis

# %%
grp = A.combo_groups(["PIK3CA", "TP53"], min_n=1, wt_label="neither").replace({"PIK3CA": "PIK3CA only", "TP53": "TP53 only", "PIK3CA+TP53": "both"})
d1 = S.join(grp, how="inner")
order4 = ["neither", "PIK3CA only", "TP53 only", "both"]
display(pd.crosstab(d1.profile, d1.subtype, margins=True))
d1x = d1[d1.profile.isin(["PIK3CA only", "TP53 only"])]
ax, t1 = cb.km_plot(d1x, "profile", order=["PIK3CA only", "TP53 only"], title="M1  OS from metastatic diagnosis: PIK3CA-only vs TP53-only", xlabel="Months from metastatic diagnosis")
h1 = cb.hr(d1x.assign(feat=(d1x.profile == "TP53 only")), "feat")
h1s = cb.hr(d1x.assign(feat=(d1x.profile == "TP53 only")), "feat", adjust=["subtype", "age_met"])
t1["HR_tp53_vs_pik3ca"] = h1["HR"]; t1["HR_adj_subtype_age"] = h1s["HR"]; t1["p_adj"] = h1s["p_cox"]
cb.save(DS, "M1", table=t1, fig=ax.figure, slug="pik3ca_only_vs_tp53_only")
display(t1)
hrp = d1x[d1x.subtype == "HER2-/HR+"]
ax, t1b = cb.km_plot(hrp, "profile", order=["PIK3CA only", "TP53 only"], title="M1  HR+/HER2− only: PIK3CA-only vs TP53-only", xlabel="Months from metastatic diagnosis")
h1b = cb.hr(hrp.assign(feat=(hrp.profile == "TP53 only")), "feat")
t1b["HR_tp53_vs_pik3ca"] = h1b["HR"]
cb.save(DS, "M1", table=t1b, fig=ax.figure, slug="hrpos_her2neg_only")
display(t1b)
ax, t1c = cb.km_plot(d1, "profile", order=order4, title="M1  all four groups", xlabel="Months from metastatic diagnosis")
cb.save(DS, "M1", table=t1c, fig=ax.figure, slug="four_groups")
cb.finding("M1", "consistent with BPC BrCa Br1" if t1.p_logrank[0] < 0.05 else "not reproduced",
           f"PIK3CA-only n={t1.n[0]} median {t1.median_months[0]} vs TP53-only n={t1.n[1]} median {t1.median_months[1]} mo from met dx, p={cb.fmt_p(t1.p_logrank[0])}, "
           f"HR {h1['HR']:.2f}, subtype/age-adjusted {h1s['HR']:.2f} (p={cb.fmt_p(h1s['p_cox'])}); within HR+/HER2−: median {t1b.median_months[0]} vs {t1b.median_months[1]} mo, HR {h1b['HR']:.2f} (p={cb.fmt_p(t1b.p_logrank[0])}).", F)

# %% [markdown]
# ## M2 — Gene screen vs OS from metastatic diagnosis (all patients, subtype-adjusted HR), and within PIK3CA-mutant patients (Br3 analogue)

# %%
scr = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel"), min_n=10, adjust=["subtype", "age_met"])
ax = cb.forest_plot(scr, top=15, title="M2  All MBC: gene features vs OS from metastatic diagnosis (HR adjusted for subtype and age)")
cb.save(DS, "M2", table=scr, fig=ax.figure, slug="gene_screen_all")
display(scr.head(15))
pik_ids = A.status("PIK3CA").pipe(lambda s: s.index[s])
scr3 = cb.gene_screen(S, A, kinds=("mut", "amp", "homdel"), min_n=10, patients=pik_ids, adjust=["subtype", "age_met"])
scr3 = scr3[scr3.gene != "PIK3CA"].reset_index(drop=True)
ax = cb.forest_plot(scr3, top=12, title="M2  Within PIK3CA-mutant MBC: co-alterations vs OS (Br3 analogue)")
cb.save(DS, "M2", table=scr3, fig=ax.figure, slug="gene_screen_in_pik3ca")
display(scr3.head(12))
hits = scr[scr.q_bh < 0.1]; hits3 = scr3[scr3.q_bh < 0.1]
cb.finding("M2", "signal" if (len(hits) or len(hits3)) else "no signal",
           "All MBC, FDR<0.1: " + (", ".join(f"{r.feature} HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits.itertuples()) or "none") +
           ". Within PIK3CA-mut, FDR<0.1: " + (", ".join(f"{r.feature} HR {r.HR:.2f} q={r.q_bh:.2g}" for r in hits3.itertuples()) or "none") + ".", F)

# %% [markdown]
# ## M3 — Biopsy timing and acquired alterations: treatment-naive vs after first-line therapy
# The study records whether the sequenced sample was taken before any first-line systemic therapy or after it. ESR1 mutations are the classic therapy-acquired alteration in HR+ disease; this is the sanity check for using this study for pre/post comparisons (cf. CRC questions C2/C3).

# %%
hrpos = S[S.subtype == "HER2-/HR+"]
timing = hrpos.biopsy_timing.replace({"gt 30": "> 30 days after first-line start", "le 30": "<= 30 days after first-line start", "Treatment naive": "treatment naive"})
anchor = (timing[timing != "not stated"] != "treatment naive")
co = A.cooccurrence(anchor, kinds=("mut", "amp", "homdel"), min_n=5)
co = co.rename(columns={"n_anchor": "n_post_treatment", "altered_in_anchor": "altered_post", "pct_in_anchor": "pct_post", "n_rest": "n_naive", "altered_in_rest": "altered_naive", "pct_in_rest": "pct_naive"}).sort_values("p_fisher")
cb.save(DS, "M3", table=co, slug="post_vs_naive_hrpos")
display(timing.value_counts())
display(co.head(12))
esr1 = co[co.gene == "ESR1"]
cb.finding("M3", "descriptive",
           f"HR+/HER2− with stated timing: {int(anchor.sum())} post-first-line vs {int((~anchor).sum())} treatment-naive samples. " +
           (f"ESR1 mut {esr1.pct_post.iloc[0]}% post vs {esr1.pct_naive.iloc[0]}% naive (Fisher p={cb.fmt_p(esr1.p_fisher.iloc[0])}). " if len(esr1) else "") +
           "Top differences: " + ", ".join(f"{r.gene} {r.kind} {r.pct_post}% vs {r.pct_naive}%" for r in co.head(3).itertuples()) + ".", F)

# %% [markdown]
# ## Summary of findings

# %%
fd = pd.DataFrame(F)
cb.save(DS, "findings", table=fd)
pd.set_option("display.max_colwidth", 400)
fd
