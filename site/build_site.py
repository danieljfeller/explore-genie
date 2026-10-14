"""Build the two-page results site into site/dist/:

  index.html      plain-language explanation
  technical.html  the expert version (for Jacob)
  site.css / site.js / data.js   shared assets; figures/<dataset>/*.png copied from results/

Inputs: results/<dataset>/findings.csv and the analysis CSVs, site/curves.json (from extract_curves.py), and the copy below.
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DIST = ROOT / "site" / "dist"
REPO = "https://github.com/danieljfeller/explore-genie"
BRANCH = "cbio-genie-analyses"

curves = json.loads((ROOT / "site" / "curves.json").read_text()) if (ROOT / "site" / "curves.json").exists() else {}

# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------
DATASETS = [
    dict(id="bpc_nsclc", tab="Lung", name="Non-small cell lung cancer", source="GENIE BPC NSCLC v2.0-public", nb="01_bpc_nsclc",
         cohort="1,846 patients, 945 deaths, median survival from diagnosis 45.7 months", questions=["N1", "N2", "N3", "N4", "N5", "N6"],
         lay="Lung cancer is where tumour DNA tests matter most, because several gene changes (EGFR, ALK, KRAS) have their own drugs. These 1,846 patients were treated at four US and Canadian hospitals and have complete records of every drug they received.",
         expert="BPC NSCLC v2.0: 1,846 patients, one index cancer each; OS from diagnosis (tt_os_dx), regimen-level OS from start (tt_os_g). Alteration calls from the earliest sequenced sample of the index cancer; panel-coverage masks applied. 107 patients have a sequencing report dated after last follow-up and drop out of the left-truncated models only."),
    dict(id="genie_akt1", tab="Breast (AKT1)", name="Breast cancer, AKT1 cohort", source="AACR GENIE AKT1 cohort (genie.cbioportal.org)", nb="02_genie_akt1",
         cohort="428 patients, 191 deaths, median survival from metastatic diagnosis 55.6 months", questions=["A2", "A1", "A3", "A4", "A5"],
         lay="A smaller study of 428 women with advanced, hormone-sensitive breast cancer, assembled to study the AKT1 gene. It records which drug classes each patient ever received, but not when they started them.",
         expert="Case-control design around AKT1 (126 mutant / 302 wild-type, almost all ER+). Exposure flags are ever-received, any line. Timeline gives metastatic-diagnosis and specimen days, so OS from metastatic diagnosis with left truncation at specimen collection is the primary endpoint; OS from initial diagnosis is shown only for the replicate. 25% of samples are from panels (MDA, GRCC) whose gene lists are not released and are assumed to cover every gene."),
    dict(id="dfci_glioma", tab="Brain", name="Glioblastoma", source="DFCI-Profile glioma cohort 2013–2018 (Touat et al., Nature 2020)", nb="03_dfci_glioma",
         cohort="913 glioblastoma patients, 575 deaths", questions=["G1", "G2"],
         lay="Glioblastoma is the most aggressive brain tumour; typical survival is under two years. This Boston study sequenced 926 of them and noted whether each sample was taken before or after the standard chemotherapy, temozolomide.",
         expert="1,535 gliomas / 1,509 patients; glioblastoma by CANCER_TYPE_DETAILED (926 samples, 913 patients after keeping one sample each, TMZ-exposed sample preferred). Per-sample exposure flags, per-patient OS from diagnosis and per-sample OS from collection (SMP_OS_MOS). No regimen table. Discrete CNA for ~500 genes on OncoPanel v1–v3."),
    dict(id="bpc_bladder", tab="Bladder", name="Bladder (urothelial) cancer", source="GENIE BPC BLADDER v1.0-public", nb="04_bpc_bladder",
         cohort="706 urothelial patients, 389 deaths, median survival from diagnosis 56.9 months", questions=["B1", "B2"],
         lay="Urothelial carcinoma is the common form of bladder cancer. These 716 patients have full treatment histories, so survival can be counted from the day each drug started.",
         expert="716 patients; urothelial carcinoma by OncoTree code (BLCA, UTUC, BLSC, BLAD …) or ICD-O-3 8120–8131, n=706. CDKN2A/B altered = deep deletion or non-synonymous mutation of either gene. Treatment comparisons use metastatic-setting regimens (started at or after first distant metastasis)."),
    dict(id="bpc_brca", tab="Breast", name="Breast cancer", source="GENIE BPC BrCa v1.0-public", nb="05_bpc_brca",
         cohort="1,130 patients, 551 deaths", questions=["Br1", "Br2", "Br3", "Br4", "Br5", "Br6"],
         lay="Breast cancer comes in biological types (hormone-sensitive, HER2-positive, triple-negative) with very different outlooks. Two genes dominate its tumour DNA: PIK3CA, common in the hormone-sensitive type, and TP53, common in the aggressive types. 1,130 patients with full treatment records.",
         expert="BPC BrCa v1.0: 1,130 patients (676 HR+/HER2−, 181 TNBC, 139 HR+/HER2+, 75 HR−/HER2+). PIK3CA and TP53 are confounded with subtype, so every PIK3CA-vs-TP53 comparison is repeated within HR+/HER2− and with subtype, stage and age in the Cox model. cBioPortal's OS-from-treatment attributes are per regimen (e.g. `Fulvestrant` vs `Fulvestrant, Palbociclib`)."),
    dict(id="bpc_crc", tab="Bowel", name="Colorectal cancer", source="GENIE BPC CRC v2.0-public", nb="06_bpc_crc",
         cohort="1,485 patients, 690 deaths, median survival from diagnosis 63.2 months", questions=["C1", "C2", "C3", "C4"],
         lay="Bowel cancer treated with the standard chemotherapy combinations (FOLFOX, FOLFIRI). The questions here were about whether tumours sampled before and after chemotherapy look different, and which of the three big bowel-cancer genes matter for survival.",
         expert="BPC CRC v2.0: 1,485 patients, 1,550 sequenced samples. Sample timing from the pathology-procedure date of each sample (dx_path_proc_cpt) relative to the first leucovorin-containing regimen (3,063 such regimens; FOLFOX 1,013). Pre/post comparison is sample-level (Fisher, BH); the paired within-patient analysis restricts to genes on both samples' panels."),
    dict(id="bpc_panc", tab="Pancreas", name="Pancreatic cancer", source="GENIE BPC PANC v1.0-public", nb="07_bpc_panc",
         cohort="1,109 patients, 879 deaths, median survival from diagnosis 22.1 months", questions=["P1"],
         lay="Pancreatic cancer is almost always driven by the KRAS gene (86% here) and has the shortest survival of any cancer in this project. The question was which additional gene changes make it worse still.",
         expert="BPC PANC v1.0: 1,109 patients; KRAS-mutant 952. Median diagnosis-to-sequencing lag 4.4 months against a median OS of 22 months, so left truncation matters more here than in any other cohort."),
    dict(id="bpc_prostate", tab="Prostate", name="Prostate cancer", source="GENIE BPC Prostate v1.0-public", nb="08_bpc_prostate",
         cohort="1,116 patients, 424 deaths, median survival from diagnosis 156.7 months", questions=["Pr1", "Pr2", "Pr3"],
         lay="Half of prostate cancers carry a rearrangement that switches on the ERG gene, which a drug company is now trying to target. The questions asked how common the change is, and what else in the tumour predicts a worse course.",
         expert="BPC Prostate v1.0: 1,116 patients. ERG-altered = any ERG structural variant, mutation, amplification or deep deletion (217 of 222 are TMPRSS2-ERG fusions). Fusion detection is panel-dependent: 0% on DFCI-ONCOPANEL-1 and the UHN panels, 27–33% on MSK-IMPACT468 / DFCI-ONCOPANEL-3 / VICC-01-T7; analyses are repeated within detecting panels."),
    dict(id="genie_v20", tab="ALK (all cancers)", name="ALK-fusion tumours across GENIE", source="AACR GENIE v20.0-public main release", nb="09_genie_v20",
         cohort="953 ALK-fusion patients (790 lung); derived survival usable for 772, 303 deaths", questions=["V1", "V2", "V3", "V4"],
         lay="GENIE's main database holds DNA tests from 242,866 patients at 19 hospitals, but no treatment records and no diagnosis dates. The questions asked whether survival can be measured at all from what it does record, and used the ALK gene fusion (a rare lung-cancer driver with its own drugs) as the test case.",
         expert="GENIE v20.0-public: 242,866 patients / 289,869 samples. Derived OS = (INT_DOD or INT_CONTACT) − AGE_AT_SEQ_REPORT_DAYS, i.e. from the sequencing report; intervals censored at >89 / <18 years are kept at the boundary and flagged. Cohort = ALK structural variant (1,152 samples, 953 patients), which reproduces the deck's 1,153-sample count; `any ALK alteration` would add 9,525 samples with passenger point mutations (only 20% lung). Screens require a gene on ≥50% of the cohort's panels and adjust for age, sex and centre."),
    dict(id="dfci_mbc", tab="Breast (DFCI)", name="Metastatic breast cancer, DFCI", source="Metastatic Breast Cancer 2013–2016 (DFCI, CCR 2020)", nb="10_dfci_mbc",
         cohort="856 patients, 271 deaths, median survival from metastatic diagnosis 63.7 months", questions=["M1", "M2", "M3"],
         lay="A second, independent breast-cancer study from Boston with no questions attached in the original plan. It is used here to check whether the breast-cancer findings above repeat in different patients.",
         expert="856 patients, one sample each, OS from metastatic diagnosis (the attribute the portal's KM uses). No regimen table and no sequencing date, so no left truncation is possible. Used as an independent replication cohort for Br1 and Br3."),
]

# ----------------------------------------------------------------------------------------------
# Questions: deck text, comparison, lay copy (class: holds / illusion / unclear / descriptive), expert take, charts
# ----------------------------------------------------------------------------------------------
Q = {
    "N1": dict(text="What is the OS of patients with EGFR mutations?", compare="EGFR-mutant vs EGFR-wild-type NSCLC; OS from diagnosis",
               lay_class="holds", lay="Patients whose lung tumour carried an EGFR change lived longer: typical survival 4.7 years against 3.5 years without it. That is what doctors expect, because EGFR changes can be treated with targeted pills, and the benefit is concentrated in the 'classic' EGFR changes those pills were designed for (median 5.3 years).",
               take="Holds after left truncation (HR 0.75) and within stage IV (HR 0.67). Classic L858R / exon-19 deletions drive it; exon-20 insertions and atypical mutations do no better than wild-type.", charts=[("nsclc_n1", None)]),
    "N2": dict(text="How is OS in EGFR+ patients affected by co-occurring mutations?", compare="Within EGFR-mutant: each co-alteration present vs absent; OS from diagnosis; FDR over 64 features",
               lay_class="holds", lay="Among those EGFR patients, a second change in the TP53 gene roughly halved the typical survival time (3.7 versus 7.6 years). TP53 was the only partner gene that passed the statistical bar; CDKN2A loss and a few others leaned the same way but could be chance.",
               take="TP53 is the only FDR-significant co-alteration (HR 1.77, q=0.007). NFKBIA amp, CDKN2A homdel and NKX2-1 amp are nominal (q 0.16–0.33) and worth a look in a larger cohort.", forest=("bpc_nsclc", "N2_comutation_screen_in_egfr.csv", "Co-alterations vs OS within EGFR-mutant NSCLC")),
    "N3": dict(text="Which mutations most commonly co-occur with EGFR, and do any convey a negative prognosis?", compare="Frequency in EGFR-mutant vs wild-type (Fisher), joined to the N2 screen",
               lay_class="descriptive", lay="TP53 is also the most common partner, found in 59% of EGFR tumours, followed by loss of the CDKN2A/B genes (11%) and extra copies of NKX2-1 (9%). Of these, only TP53 clearly worsened the outlook.",
               take="Co-occurrence ranking is TP53 mut 59%, CDKN2A/B homdel 11%, NKX2-1 amp 9%, KMT2D mut 8%; only TP53 is prognostic at FDR<0.1. NKX2-1 amp is enriched in EGFR-mutant tumours (9.3% vs 3.9%, Fisher p=1e-4)."),
    "N4": dict(text="Replicate: in EGFR+ patients osimertinib clearly outperforms chemotherapy", compare="EGFR-mutant: osimertinib regimen vs chemotherapy regimen; OS from diagnosis vs from regimen start",
               lay_class="illusion", lay="The slide suggested osimertinib beat chemotherapy by a wide margin. Measured properly, from the day each treatment started, the two groups look the same (21 versus 26 months). The original gap appears only when you count from diagnosis, and it comes from the fact that most osimertinib patients had already survived an earlier treatment before they got the drug.",
               take="From diagnosis the received-osimertinib group has median 67.5 vs 37.7 months (p=6e-4) because 91% got osimertinib at 2nd+ line. From regimen start in the metastatic setting: 20.8 vs 25.5 months, HR (LT) 0.92, p=0.2. First-line only is n=11 vs 68. Not a treatment effect in these data.",
               charts=[("nsclc_n4_dx", "From diagnosis (deck view)"), ("nsclc_n4_reg", "From the start of treatment")]),
    "N5": dict(text="Can we identify mutation profiles of EGFR+ patients who respond well vs poorly to osimertinib?", compare="Good (≥12 months without a next treatment) vs poor (next treatment or death <6 months); Fisher per gene; time-to-next-treatment and PFS screens",
               lay_class="unclear", lay="No. Sixty patients did well on osimertinib and 48 did poorly, and no gene pattern separated them beyond what chance would produce. A bigger study would be needed.",
               take="Response had to be defined by proxy: BPC PFS counts any imaging or oncologist note of progression (median 5 months on osimertinib), so time to next treatment was used (good n=60, poor n=48). No feature passes FDR<0.1 in the profile comparison, the TTNT screen or the PFS-I screen; nominal top is NKX2-1 amp (TTNT HR 1.9, p=0.01). The response definition is an open decision.",
               forest=("bpc_nsclc", "N5_ttnt_screen_on_osimertinib.csv", "Time to next treatment on osimertinib, by baseline co-alteration")),
    "N6": dict(text="Compare EGFR vs KEAP1 vs STK11 vs KEAP1+STK11: OS and response to different treatments", compare="Mutually exclusive profiles (overlap kept as its own group); OS from diagnosis; OS from start of immunotherapy and platinum chemotherapy",
               lay_class="holds", lay="Tumours with changes in both KEAP1 and STK11 had the worst outlook: typical survival 2.5 years, against 4.3 years for tumours with none of the three genes changed. These patients also did especially poorly on immunotherapy (6 months versus 13), which matches what oncologists have reported elsewhere.",
               take="Adjusted HR vs none: EGFR 0.74, KEAP1 1.40, STK11 1.31, KEAP1+STK11 2.02 (p=4e-7). On immunotherapy-containing regimens the KEAP1+STK11 median is 6.0 months vs 12.9 for none (n=35 vs 264, log-rank across profiles p=0.25). The strongest replicable signal in the lung data.",
               charts=[("nsclc_n6", "From diagnosis"), ("nsclc_n6_io", "From start of immunotherapy")]),
    "A2": dict(text="Replicate: patients who received aromatase therapy lived longer than those who did not", compare="AI ever received vs not; OS from initial diagnosis (deck) and from metastatic diagnosis in ER+ patients",
               lay_class="illusion", lay="Patients who ever took an aromatase inhibitor lived about twice as long. But you have to live long enough to receive a drug for it to be recorded, and the untreated group includes patients whose tumours were not hormone-sensitive in the first place. The difference persists after the obvious corrections, but a study without treatment dates cannot say whether the drug caused it.",
               take="Naive HR 0.32; ER+ only, from metastatic diagnosis, left-truncated at specimen, age/HER2-adjusted HR 0.33. The untreated group is n=58 and 19% ER−. Ever-received exposure without start dates leaves the immortal-time component unquantifiable; this cohort cannot support a treatment-effect claim.",
               charts=[("akt1_a2_init", "From initial diagnosis (deck view)"), ("akt1_a2_met", "ER+ patients, from metastatic diagnosis")]),
    "A1": dict(text="In patients receiving aromatase therapy, are specific mutations associated with worse OS?", compare="Within AI-treated: each gene feature vs OS from metastatic diagnosis; FDR over 32 features",
               lay_class="holds", lay="Among treated patients only one gene stood out: a TP53 change went with shorter survival (2.8 versus 5.5 years from the cancer's spread). No other gene passed the bar.",
               take="TP53 mut HR 2.46 (q=2e-7) is the only hit; NBN amp and CCND1 amp are nominal. AKT1 is the cohort's design variable and is null (HR 0.95).",
               charts=[("akt1_a1_tp53", None)], forest=("genie_akt1", "A1_gene_screen_in_ai_treated.csv", "AI-treated: gene features vs OS from metastatic diagnosis")),
    "A3": dict(text="Do AI-treated patients with PIK3CA mutations do better or worse than those without? Ask this for each mutation", compare="Within AI-treated: PIK3CA, AKT1, TP53, ESR1, GATA3, CDH1, MAP3K1 mut vs wt",
               lay_class="unclear", lay="PIK3CA changes made no measurable difference (5.1 versus 4.7 years), and neither did AKT1 or ESR1. The 'each mutation' part of the question is answered by the screen above: only TP53.",
               take="PIK3CA HR 0.75 [0.52–1.07], p=0.25; AKT1 0.95; ESR1 0.69 (p=0.27); TP53 2.46. Underpowered for anything but large effects."),
    "A4": dict(text="How do AI-treated patients perform with co-occurring PIK3CA + AKT1 + TP53 mutations?", compare="Mutually exclusive profiles over the three genes; pairwise combinations vs the rest",
               lay_class="holds", lay="Patients with both AKT1 and TP53 changes did worse (2.8 versus 5.2 years); PIK3CA plus TP53 leaned the same way. Only five patients had PIK3CA and AKT1 together and two had all three, too few to say anything.",
               take="AKT1+TP53 HR 2.06 (p=0.001, n=28); PIK3CA+TP53 HR 1.38 (p=0.09, n=29); triple n=2. The TP53 effect carries the combinations."),
    "A5": dict(text="126 patients received CDK4/6 inhibitors, 123 concurrently with an AI: survival for the combination, then which mutations predict poorer outcomes", compare="AI + CDK4/6i treated: OS from metastatic diagnosis and time on the CDK4/6 inhibitor; gene screens with ≥8 per group",
               lay_class="unclear", lay="Among the 123 patients on an aromatase inhibitor plus a CDK4/6 inhibitor, typical survival was 8 years from the cancer's spread, but they stayed on the CDK4/6 drug only 5.6 months on average, and no gene predicted either outcome.",
               take="Median OS from metastatic dx 95.7 months (29 deaths); median time on CDK4/6i 5.6 months (116 discontinuations). No feature passes FDR<0.1 on either endpoint; nominal: TP53 mut (OS HR 2.1, p=0.01), NF1 mut (discontinuation HR 1.8, p=0.06). Reference PMID 39754979 is the comparator."),
    "G1": dict(text="Replicate: among glioblastoma, 135 patients who received temozolomide before resection had higher OS (log-rank p = 1.24e-8 in deck)", compare="GBM: temozolomide before the sampled resection yes vs no; OS from diagnosis vs from sample collection",
               lay_class="illusion", lay="The slide showed that patients who had temozolomide before their operation lived much longer. But every one of those 135 samples came from a second operation after the tumour returned, so those patients had already survived long enough to need one. Counting from the operation instead, they did worse (11 versus 20 months), as you would expect for recurrent disease.",
               take="Replicates (p=6.5e-10 vs deck 1.2e-8; naive HR 0.49), but 135/135 TMZ-exposed samples are recurrence resections. From sample collection: 11.5 vs 19.5 months (p=1e-13); left-truncated, IDH- and age-adjusted HR 1.98; recurrence-only, from sample, HR 3.09. Textbook immortal time.",
               charts=[("glioma_g1_dx", "From diagnosis (deck view)"), ("glioma_g1_sample", "From the date of the operation")]),
    "G2": dict(text="How does each CNA affect OS within each TMZ group? Do patients with CDKN2A/CDKN2B and MTAP deletions do worse?", compare="GBM, by TMZ group: 11 copy-number features; OS from diagnosis and from sample; IDH- and age-adjusted HR",
               lay_class="holds", lay="Loss of the CDKN2A gene went with slightly shorter survival (21 versus 25 months) in first-surgery tumours; loss of MTAP did not. Extra copies of PDGFRA or MDM4 also tracked with shorter survival. Within the small group sampled after temozolomide, nothing held up except extra copies of EGFR.",
               take="All GBM: CDKN2A homdel 40%, adjusted HR 1.22 (p=0.02); MTAP homdel 14%, HR 0.94 (null); co-deletion HR 0.94. Within-stratum FDR<0.1: no-TMZ PDGFRA amp HR 1.54, MDM4/PIK3C2B amp 1.42, CDKN2A homdel 1.13; TMZ stratum EGFR amp HR 1.30. The MTAP-deletion hypothesis is not supported here.",
               charts=[("glioma_g2_cdkn2a", None)], forest=("dfci_glioma", "G2_cna_by_tmz_group.csv", "Copy-number features vs OS, all GBM (IDH- and age-adjusted)", "all GBM")),
    "B1": dict(text="How do bladder urothelial carcinoma patients do with CDKN2A or CDKN2B mutations?", compare="Urothelial: CDKN2A/B altered (deep deletion or mutation) vs intact; OS from diagnosis; stage/age-adjusted HR",
               lay_class="holds", lay="Loss of the CDKN2A/B genes cut typical survival roughly in half (3 years versus 6), and the gap holds after accounting for how advanced the cancer was at diagnosis. This is the clearest new finding in the whole project.",
               take="n=187 altered: median 36.2 vs 70.3 months, log-rank p=6e-5, stage/age-adjusted left-truncated HR 1.75 [1.41–2.18]; stage IV only HR 1.77. Deep deletion carries it (HR 1.99); CDKN2A point mutations are null (n=40).",
               charts=[("bladder_b1", None)]),
    "B2": dict(text="How is OS affected by different treatments or combinations, especially in the CDKN2A/B group?", compare="Metastatic-setting regimens by class; OS from regimen start; CDKN2A/B altered vs intact within each class",
               lay_class="descriptive", lay="The disadvantage shows up on every treatment type: platinum chemotherapy, immunotherapy, gemcitabine. From the start of treatment, patients on cisplatin lived longest (30 months) and those on taxanes or immunotherapy, given later in the course, the shortest (11 to 14 months). Treatments given at different stages cannot be ranked against each other from this.",
               take="Within-class HRs for CDKN2A/B altered: platinum 1.73 (p=4e-5), cisplatin 1.68, immunotherapy 1.47 (p=0.02), gemcitabine 1.59; taxane 0.91 (null). Medians across classes reflect line of therapy (cisplatin 83% first line, immunotherapy 11%).",
               charts=[("bladder_b2_platinum", None)]),
    "Br1": dict(text="Replicate: PIK3CA-mutant patients have significantly better OS than TP53-mutant patients", compare="PIK3CA-only vs TP53-only (overlap dropped, as cBioPortal does); OS from diagnosis; subtype-adjusted; within HR+/HER2−",
                lay_class="holds", lay="Tumours with a PIK3CA change had a much better outlook than those with a TP53 change: typical survival 11 years versus 5.4. Part of this is because the two changes occur in different breast-cancer types, but the gap remains large within the common hormone-sensitive type (10 versus 6 years), so it is not just a type effect.",
                take="HR (LT) TP53-only vs PIK3CA-only 2.15; adjusted for subtype, stage and age 2.65 (p=7e-10); within HR+/HER2− 3.03 (p=5e-10). Replicated and robust; reproduces in the independent DFCI MBC cohort (M1).",
                charts=[("brca_br1", "All subtypes, four groups"), ("brca_br1_hrpos", "HR+/HER2− only")]),
    "Br2": dict(text="Are there treatments that improved OS in TP53-mutant patients?", compare="By treatment class (metastatic setting): TP53-mutant vs wild-type from regimen start",
                lay_class="descriptive", lay="Patients with TP53 changes did worse on every type of treatment. The gap was smallest, and not statistically clear, on HER2-targeted drugs and everolimus; it was largest on CDK4/6 inhibitors and fulvestrant. Nothing in this data points to a treatment that cancels the TP53 disadvantage.",
                take="TP53-mut vs wt HR by class: CDK4/6i 2.24, fulvestrant 2.41, tamoxifen 2.28, capecitabine 1.62, eribulin 1.47, AI 1.63, chemotherapy 1.41 (ns), HER2-targeted 1.52 (ns), everolimus 1.55 (ns). Classes are not comparable to each other (line and subtype differ).",
                forest=("bpc_brca", "Br2_tp53_by_treatment_class.csv", "TP53-mutant vs wild-type, by treatment class (OS from regimen start)")),
    "Br3": dict(text="Are there co-mutations in PIK3CA patients that caused worse outcomes?", compare="Within PIK3CA-mutant: each co-alteration vs OS from diagnosis; FDR over 45 features",
                lay_class="holds", lay="Yes, one: among PIK3CA patients, a second change in TP53 halved survival (5.5 versus 11 years). Extra copies of RAD21 leaned the same way but could be chance.",
                take="TP53 mut HR 2.62, subtype-adjusted 3.83, q=3e-4; RAD21 amp nominal (HR 2.6, q=0.19). Same hit reproduces in DFCI MBC (M2).",
                forest=("bpc_brca", "Br3_comutation_screen_in_pik3ca.csv", "Co-alterations vs OS within PIK3CA-mutant breast cancer")),
    "Br4": dict(text="Break down OS from the start of different treatments", compare="All patients, metastatic-setting regimens, first regimen of each class; OS from regimen start",
                lay_class="descriptive", lay="From the day a treatment started, survival ran from 6.8 years on HER2-targeted drugs and 6.3 on tamoxifen, through 4.3 years on aromatase inhibitors, down to 9 months on eribulin and 6 on immunotherapy. The order mostly reflects when in the illness each drug is used, not how good the drug is.",
                take="Medians from regimen start: HER2-targeted 80.9, tamoxifen 75.1, AI 51.9, chemotherapy 45.8, fulvestrant 32.5, CDK4/6i 32.5, everolimus 29.1, capecitabine 24.0, eribulin 8.7, immunotherapy 6.4 months. First-line share ranges from 54% (chemotherapy) to 0% (everolimus, eribulin)."),
    "Br5": dict(text="Replicate: PIK3CA-mut vs TP53-mut patients who received fulvestrant show no significant OS difference", compare="Fulvestrant monotherapy regimen (cBioPortal's attribute), PIK3CA-only vs TP53-only, OS from regimen start; then any fulvestrant-containing regimen",
                lay_class="unclear", lay="On fulvestrant alone the PIK3CA-versus-TP53 difference was not statistically clear, matching the slide, but only 50 patients got fulvestrant by itself. Counting everyone who got fulvestrant in any combination (123 patients), the PIK3CA advantage is clear again.",
                take="Monotherapy regimen: n=35 vs 15, median 47.9 vs 28.6 months, p=0.30, HR 2.13 [0.89–5.1]. Any fulvestrant-containing regimen: n=77 vs 46, 43.4 vs 22.9 months, p=2e-4, HR 2.72. The deck's null is a power problem, not a biological one.",
                charts=[("brca_br5", None)]),
    "Br6": dict(text="Replicate: on fulvestrant + palbociclib, PIK3CA-mut OS was better than TP53-mut", compare="`Fulvestrant, Palbociclib` regimen, PIK3CA-only vs TP53-only, OS from regimen start",
                lay_class="holds", lay="On the combination of fulvestrant and palbociclib, patients with PIK3CA changes lived longer than those with TP53 changes (36 versus 23 months). The slide holds, in a small group.",
                take="n=32 vs 17, median 36.4 vs 23.0 months, p=0.001, HR (LT) 3.17 [1.49–6.75], subtype/age-adjusted 4.22. Replicated.",
                charts=[("brca_br6", None)]),
    "C1": dict(text="Replicate: OS of patients biopsied before leucovorin vs after leucovorin was very different", compare="Biopsy before vs after the first leucovorin regimen; OS from diagnosis vs from the biopsy date",
               lay_class="illusion", lay="Patients biopsied after chemotherapy appeared to live 20 months longer, but a second biopsy after treatment can only happen in someone who survived the treatment. Counted from the biopsy date, there is no difference at all (48 versus 45 months).",
               take="From diagnosis: 52.3 vs 72.6 months, p=2e-10, naive HR 0.61. From sample collection: 47.6 vs 44.7 months, p=0.64; left-truncated, stage/age-adjusted HR 0.81 (p=0.02; residual selection of patients fit for metastasectomy). Same structure as G1.",
               charts=[("crc_c1_dx", "From diagnosis (deck view)"), ("crc_c1_sample", "From the date of the biopsy")]),
    "C2": dict(text="How do mutational profiles differ in samples taken before vs after chemotherapy?", compare="Sample-level, all 1,550 samples, pre vs post first chemotherapy (and leucovorin) regimen; Fisher per feature, FDR",
               lay_class="unclear", lay="Overall, tumours sampled after chemotherapy did not show a different pattern of gene changes. A few genes were rarer or commoner, but none beyond what chance would give in a search of 350 genes.",
               take="No feature passes FDR<0.1 for pre vs post chemotherapy (503 vs 874 samples); nominal BCL2 homdel 0.4%→2.1%, PIK3CA mut 22%→16%. Against leucovorin specifically, CSF3R mut 0.4%→3.7% reaches q=0.04. Samples from the same patient are not independent; see C3."),
    "C3": dict(text="How many patients have both pre and post biopsies, and how did their mutations change?", compare="Paired pre/post-chemotherapy samples within patient; genes on both panels",
               lay_class="descriptive", lay="Only 26 of the 1,485 patients had a biopsy both before and after chemotherapy. Between the two, their tumours gained 50 mutations and lost 37, with no single gene standing out (TP53 was gained in four patients). Too few pairs to conclude anything.",
               take="26 paired patients; mutations kept 98, gained 50, lost 37; amplifications gained 31, lost 44. Most often gained: TP53 (4), AURKA amp (3), CDKN2A/B homdel (2 each). Panel differences between the two samples are handled by restricting to genes on both."),
    "C4": dict(text="How does mutation status affect survival: KRAS vs TP53 vs APC, plus co-mutations", compare="Single genes (overlapping) and mutually exclusive KRAS/TP53/APC profiles; OS from diagnosis; stage/age-adjusted",
               lay_class="holds", lay="KRAS changes went with shorter survival (4.5 versus 5.7 years) and APC changes with longer; TP53, the most common change, made no difference. Combinations of the three added nothing beyond the KRAS effect.",
               take="Adjusted HR: KRAS 1.44, APC 0.76, TP53 0.98, BRAF 1.03, PIK3CA 0.79. Profiles vs none of the three: KRAS+TP53 1.30 (ns), KRAS+TP53+APC 1.01. APC's protective association is the classic-pathway vs hypermutated/serrated confounding, not a target.",
               charts=[("crc_c4", None)]),
    "P1": dict(text="How is OS of KRAS-mutant patients further differentiated by co-mutations? Which co-mutations associate with worse outcome?", compare="Within KRAS-mutant: each co-alteration vs OS from diagnosis; CDKN2A/B and MTAP deletions individually",
               lay_class="holds", lay="In KRAS-driven pancreatic cancer, losing the CDKN2A/B genes (15 versus 21 months) or the SMAD4 gene (11 versus 20 months) predicted shorter survival. MTAP loss leaned the same way but was too rare to be sure.",
               take="FDR<0.1: CDKN2A homdel HR 1.47, CDKN2B homdel 1.53, SMAD4 homdel 1.66, NTRK3 mut 2.3 (n=15), plus CDKN2A mut 1.29, TP53 mut 1.26, AKT2 amp 1.53. MTAP homdel HR 1.4 (n=27, p=0.11); CDKN2A+MTAP co-deletion adjusted HR as reported in the notebook.",
               charts=[("panc_p1_cdkn2a", "CDKN2A deep deletion"), ("panc_p1_smad4", "SMAD4 deep deletion"), ("panc_p1_kras", "KRAS vs wild-type (reference)")],
               forest=("bpc_panc", "P1_comutation_screen_in_kras.csv", "Co-alterations vs OS within KRAS-mutant pancreatic cancer")),
    "Pr1": dict(text="Characterise ERG alterations (context: Parabilis is developing an ERG degrader)", compare="ERG-altered vs not; frequency by alteration kind, fusion partner and panel; OS from diagnosis",
                lay_class="descriptive", lay="An ERG change was found in 21% of patients, almost always as a fusion with the TMPRSS2 gene. But how often it is found depends heavily on which DNA test the hospital used, from 0% on older tests to 33% on newer ones, so the true rate is higher than 21%. Having the change did not by itself alter survival.",
                take="222/1,040 altered (217 fusions, 243 of 245 partnered with TMPRSS2). Fusion rate by panel: 0% (DFCI-ONCOPANEL-1, UHN), 9% (ONCOPANEL-2), 23–33% (MSK-IMPACT468, ONCOPANEL-3, VICC-01-T7). OS in detecting panels: adjusted HR 1.05 (null). Any ERG-targeting framing must restrict to panels that call the rearrangement."),
    "Pr2": dict(text="Replicate: among ERG-altered patients, AR vs PTEN co-alteration shows no significant OS difference", compare="Within ERG-altered: AR amp/mut vs PTEN loss/mut vs neither vs both; OS from diagnosis",
                lay_class="unclear", lay="Among ERG patients, those with extra copies of the AR gene and those with PTEN loss had similar survival (12 versus 7.5 years), a difference that fell just short of statistical significance. Both groups did worse than patients with neither change.",
                take="AR n=33 median 144.6 vs PTEN n=46 median 90.2 months, p=0.056, HR (LT) PTEN vs AR 0.43 [reflecting that AR amplification marks castration resistance acquired late, i.e. late sequencing]. Four-group p=8e-4 driven by the 'neither' group. Replicates as non-significant.",
                charts=[("prostate_pr2", None)]),
    "Pr3": dict(text="Is there a way to sift the data for mutational profiles that convey higher risk in ERG+ patients?", compare="Within ERG-altered: every co-alteration vs OS; repeated in ERG-wild-type to separate ERG-specific from general markers",
                lay_class="holds", lay="Within ERG tumours, a TP53 change (8 versus 15.6 years) or loss of PTEN (8 versus 13 years) marked a worse course. PTEN loss and KMT2D changes were specific to ERG tumours; TP53 is bad news in prostate cancer generally.",
                take="FDR<0.1 within ERG+: TP53 mut HR 2.73, PTEN homdel 2.43, KMT2D mut 1.72, AR amp 4.04. PTEN homdel and KMT2D mut are not significant in ERG-wild-type patients (ERG-specific); TP53 and AR amp are prognostic in both.",
                charts=[("prostate_pr3_tp53", None)], forest=("bpc_prostate", "Pr3_screen_in_erg_altered.csv", "Co-alterations vs OS within ERG-altered prostate cancer")),
    "V1": dict(text="Can OS be derived at all? Proposed: OS = (interval DOB→DOD or last contact) − (age at sequencing). Can diagnosis be inferred from age at sequencing?", compare="Feasibility of a derived endpoint from the main release",
               lay_class="descriptive", lay="Yes, partly. GENIE records each patient's age at the DNA test and the age at death or last contact, so survival can be counted from the test for 772 of the 953 ALK-fusion patients. The date of diagnosis is not recorded and cannot be worked out from the test date; in the lung dataset that does record both, the gap between them ranges from a month to several years.",
               take="Derived OS = interval to death/last contact − age at sequencing report (days). Usable for 772/953 ALK-fusion patients (vital status unreleased for 70; last contact before report for 111; 20 censored at an age boundary). Median derived OS 71 months (72 for NSCLC). Diagnosis-to-sequencing lag in BPC NSCLC ALK-fusion patients: median 3.4 months, IQR 1.6–18.",
               charts=[("v20_v1", None)]),
    "V2": dict(text="1,274 ALK-altered patients (predominantly lung): how does OS change with additional mutations?", compare="ALK-fusion NSCLC: each co-alteration vs derived OS from sequencing; genes on ≥50% of panels; age-, sex-, centre-adjusted; FDR",
               lay_class="holds", lay="In ALK-fusion lung cancer, additional changes in TP53 (3.3 versus 6.7 years), KMT2D, or loss of CDKN2A/B or MTAP went with shorter survival. These are the same 'bad partner' genes that recur across the whole project.",
               take="FDR<0.1 (n=537): KMT2D mut HR 2.21, TP53 mut 1.69, EGFR mut 2.87 (n=13), NF2 mut 3.48 (n=13), CDKN2A homdel 1.74, CDKN2B homdel 1.63, MTAP homdel 2.60, ROS1 mut 2.63 (n=12). Treat the n≈13 hits as leads only. The first (any-ALK-alteration) run produced panel artefacts; the fusion definition and the ≥50%-panel rule removed them.",
               charts=[("v20_v2_tp53", "TP53 mutation"), ("v20_v2_cdkn2a", "CDKN2A deep deletion")], forest=("genie_v20", "V2_comutation_screen.csv", "Co-alterations vs derived OS in ALK-fusion NSCLC")),
    "V3": dict(text="Extract all mutations, structural variants and CNAs for every ALK+ patient and relate profiles to survival", compare="Per-patient profile table; oncoprint of the top 25 co-alterations; OS by number of co-alterations",
               lay_class="descriptive", lay="A full list of gene changes was written out for 626 ALK-fusion lung-cancer patients. The most common partners are TP53 (30%) and loss of CDKN2A/B (20%). The more extra changes a tumour carried, the shorter the survival.",
               take="Profiles in results/genie_v20/V3_per_patient_profiles.csv. Top co-alterations: TP53 mut 30%, CDKN2A homdel 22%, CDKN2B homdel 20%, SETD2 mut 8%, MTAP homdel 8%, KMT2D mut 7%. The deck's example profile (EML4-ALK + TP53 + ERBB2 amp) occurs in 4 patients. Co-alteration burden vs OS: log-rank p=3e-6."),
    "V4": dict(text="No treatment data in v20; can treatments be identified another way?", compare="Feasibility: BPC linkage, resistance-mutation footprint, sample context",
               lay_class="descriptive", lay="Treatment is not recorded, but there are workarounds. Forty-eight tumours carry mutations that only arise after ALK drugs, so those patients were certainly treated (and did worse, as expected for resistant disease). And 49 of the 53 ALK-fusion patients in the smaller lung dataset also appear in the big database with full treatment records.",
               take="48/626 ALK-fusion NSCLC samples carry a known ALK-TKI resistance mutation (G1202R 17, L1196M 12 …): derived OS HR 2.56. 49/53 BPC NSCLC ALK-fusion patients link to v20 by GENIE ID, 41 with an ALK TKI regimen. Sequencing year and sample type give era context. The Guardant-style ~300-patient design is feasible via BPC linkage plus the footprint.",
               charts=[("v20_v4", None)]),
    "M1": dict(text="Independent replication of Br1: PIK3CA-only vs TP53-only, OS from metastatic diagnosis", compare="PIK3CA-only vs TP53-only; subtype/age-adjusted; within HR+/HER2−",
               lay_class="holds", lay="In this separate group of 856 patients the same pattern appears: PIK3CA tumours 8.8 years of survival after the cancer spread, TP53 tumours 2.5 years, and the gap holds within the hormone-sensitive type.",
               take="HR 3.01, subtype/age-adjusted 2.87 (p=5e-7); within HR+/HER2− HR 2.62 (p=7e-5). Consistent with Br1.", charts=[("mbc_m1", None)]),
    "M2": dict(text="Gene screen vs OS from metastatic diagnosis, all patients and within PIK3CA-mutant", compare="Subtype- and age-adjusted HR per feature; FDR",
               lay_class="holds", lay="Across all patients, TP53 and RB1 changes went with shorter survival and extra copies of ERBB2 (the HER2 gene, which has its own drugs) with longer. Within PIK3CA tumours, TP53 was again the bad partner.",
               take="All MBC FDR<0.1: TP53 mut HR 2.58, RB1 mut 2.24, NOTCH2 amp 1.53, ERBB2 amp 0.57, IKZF3 amp 0.59 (the ERBB2 amplicon). Within PIK3CA-mut: TP53 mut 2.65, MYC amp 2.4, RAD21/EXT1 amp (8q amplicon) ~4.",
               forest=("dfci_mbc", "M2_gene_screen_all.csv", "Gene features vs OS from metastatic diagnosis, all MBC (subtype- and age-adjusted)")),
    "M3": dict(text="Biopsy timing and acquired alterations: treatment-naive vs after first-line therapy (HR+/HER2−)", compare="Fisher per feature, post-first-line vs treatment-naive samples",
               lay_class="descriptive", lay="Samples taken after hormone therapy had ESR1 changes five times as often as untreated ones (27% versus 6%). ESR1 changes are a known way tumours escape hormone therapy, so this is a sanity check that the data behave as expected.",
               take="33 post-first-line vs 90 naive HR+/HER2− samples: ESR1 mut 27.3% vs 5.6% (Fisher p=0.002, q=0.06); PRKDC and ATM mutations also enriched. Confirms the dataset can support pre/post comparisons of the C2 kind."),
}

LAY_CLASS = {
    "holds": ("Holds up", "A real pattern in these patients, after the usual checks."),
    "illusion": ("An illusion", "The slide's result comes from how survival was counted, not from the treatment."),
    "unclear": ("No clear answer", "Too few patients, or nothing beyond chance."),
    "descriptive": ("Description", "A count or ranking, not a yes/no test."),
}


def verdict_class(v: str) -> str:
    v = v.lower()
    if "immortal" in v or "survivor" in v or "not supported" in v:
        return "artefact"
    if v.startswith("replicated") or v.startswith("confirmed") or v.startswith("consistent"):
        return "replicated"
    if v.startswith("signal"):
        return "signal"
    if v.startswith("no "):
        return "null"
    return "descriptive"


# ----------------------------------------------------------------------------------------------
# Load results
# ----------------------------------------------------------------------------------------------
findings = {}
for ds in DATASETS:
    f = RES / ds["id"] / "findings.csv"
    if f.exists():
        for r in pd.read_csv(f).itertuples():
            findings[r.question] = dict(verdict=str(r.verdict), note=str(r.note))


def forest_rows(ds_id, fname, stratum=None, top=15):
    f = RES / ds_id / fname
    if not f.exists():
        return []
    df = pd.read_csv(f)
    if stratum is not None and "stratum" in df.columns:
        df = df[df.stratum == stratum]
    if "q_bh" not in df.columns and "p_logrank" in df.columns:
        from statsmodels.stats.multitest import multipletests
        df = df.assign(q_bh=multipletests(df.p_logrank, method="fdr_bh")[1])
    df = df.dropna(subset=["HR"]).sort_values("p_logrank").head(top)
    rows = []
    for r in df.itertuples():
        rows.append(dict(feature=str(r.feature), n_alt=int(getattr(r, "n_alt", 0) or 0), n_wt=int(getattr(r, "n_wt", 0) or 0) if hasattr(r, "n_wt") else None,
                         median_alt=_num(getattr(r, "median_alt", getattr(r, "median_mut", None))), median_wt=_num(getattr(r, "median_wt", None)),
                         HR=_num(r.HR), lo=_num(r.HR_lo), hi=_num(r.HR_hi), p=_num(r.p_logrank), q=_num(r.q_bh)))
    return rows


def _num(x):
    try:
        x = float(x)
        return None if x != x or x in (float("inf"), float("-inf")) else round(x, 4)
    except Exception:
        return None


def figures_for(ds_id, qid):
    d = RES / ds_id
    return sorted(p.name for p in d.glob(f"{qid}_*.png"))


# ----------------------------------------------------------------------------------------------
# Assemble data.js
# ----------------------------------------------------------------------------------------------
data = dict(datasets=[], curves={}, forests={})
for ds in DATASETS:
    qs = []
    for qid in ds["questions"]:
        q = Q[qid]
        fd = findings.get(qid, dict(verdict="not run", note=""))
        entry = dict(id=qid, text=q["text"], compare=q["compare"], lay=q["lay"], lay_class=q["lay_class"], take=q.get("take", ""),
                     verdict=fd["verdict"], vclass=verdict_class(fd["verdict"]), note=fd["note"], figures=figures_for(ds["id"], qid),
                     charts=[c for c in q.get("charts", []) if c[0] in curves], forest=None)
        for key, _ in entry["charts"]:
            data["curves"][key] = curves[key]
        if "forest" in q:
            fspec = q["forest"]
            rows = forest_rows(fspec[0], fspec[1], stratum=fspec[3] if len(fspec) > 3 else None)
            if rows:
                fkey = f"{qid}_forest"
                data["forests"][fkey] = dict(title=fspec[2], rows=rows)
                entry["forest"] = fkey
        qs.append(entry)
    data["datasets"].append(dict(id=ds["id"], tab=ds["tab"], name=ds["name"], source=ds["source"], nb=ds["nb"], cohort=ds["cohort"],
                                 lay=ds["lay"], expert=ds["expert"], questions=qs, cohort_fig=f"cohort_gene_frequency.png" if (RES / ds["id"] / "cohort_gene_frequency.png").exists() else None))

# headline counts over the 33 deck questions
deck_q = [q for ds in data["datasets"] if ds["id"] != "dfci_mbc" for q in ds["questions"]]
counts = {k: sum(1 for q in deck_q if q["lay_class"] == k) for k in LAY_CLASS}
data["counts"] = counts
data["repo"] = REPO
data["branch"] = BRANCH

# ----------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------
HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&display=swap">
<link rel="stylesheet" href="site.css">
</head>
<body class="{cls}">
<header class="topbar">
  <div class="wrap bar">
    <a class="brand" href="index.html">Cancer genes and survival</a>
    <nav class="switch" aria-label="Audience">
      <a href="index.html" {a1}>Plain language</a>
      <a href="technical.html" {a2}>Technical</a>
    </nav>
  </div>
</header>
<main class="wrap">
"""

FOOT = """
</main>
<footer class="wrap foot">
  <p>Data: <a href="https://www.aacr.org/professionals/research/aacr-project-genie/">AACR Project GENIE</a> (BPC cohorts and the v20.0-public release, via Synapse) and
  <a href="https://genie.cbioportal.org/">genie.cbioportal.org</a> (AKT1, DFCI glioma and DFCI metastatic breast studies). Analysis code, notebooks and tables:
  <a href="{repo}/tree/{branch}">explore-genie</a>. Built by Daniel Feller with Claude Code, October 2026.</p>
  <p>These are statistical associations in past patients from a handful of academic hospitals. They are not predictions for any individual and not treatment advice.</p>
</footer>
<div id="lightbox" class="lightbox" hidden><button class="lb-close" aria-label="Close">×</button><img alt=""><div class="lb-cap"></div></div>
<script src="data.js"></script>
<script src="site.js"></script>
<script>{boot}</script>
</body>
</html>
"""

LAY_BODY = """
<section class="hero">
  <p class="eyebrow">An audit of 33 questions about tumour DNA and survival</p>
  <h1>What the tumour records say, and what they only seem to say</h1>
  <p class="lede">Nineteen hospitals have pooled the DNA tests of a quarter of a million tumours into a public project called GENIE. For some of those patients, nurses and doctors also went through the charts and recorded every treatment and how long each person lived. We used that data to test 33 questions of one form: does a particular change in a tumour's genes go with living longer or shorter?</p>
  <div class="tiles" id="tiles"></div>
  <p class="small">Counts cover the 33 questions in the original plan. Three bonus checks on an extra dataset are included below.</p>
</section>

<section>
  <h2>How to read the curves on this page</h2>
  <p>Every chart here is a <strong>survival curve</strong>. Start at the left: 100% of a group is alive. Move right through the months and the line steps down each time someone dies. Small tick marks are patients who were still alive when the records end. Where a line crosses 50% is the group's <strong>typical (median) survival</strong>. Hover or tap on a chart to read off the exact numbers.</p>
  <p>Below, lung-cancer patients with and without a change in the EGFR gene. The EGFR line stays higher the whole way: more of those patients were alive at every point.</p>
  <div class="chart" data-km='[["nsclc_n1",null]]'></div>
  <p class="small">The <em>p-value</em> in the corner is the chance of a gap at least this large arising if the two groups were really the same. Below 0.05 is the usual bar; below 0.001 is convincing.</p>
</section>

<section>
  <h2>The trap that fooled several of the original slides</h2>
  <p>Some questions compared patients who had a treatment, or a second biopsy, with patients who did not. That sounds fair but hides a trap: to be in the treated group you first have to live long enough to get the treatment. Counting survival from the original diagnosis then gives the treated group a head start that has nothing to do with the treatment. Statisticians call it <strong>immortal time</strong>.</p>
  <p>Brain-tumour patients whose sample was taken after temozolomide seem to live twice as long. Switch to counting from the day the sample was taken and the picture reverses: these are patients whose tumour came back, and they did worse.</p>
  <div class="chart" data-km='[["glioma_g1_dx","Counted from diagnosis (the slide)"],["glioma_g1_sample","Counted from the operation"]]'></div>
  <p>Four of the six slide results marked "replicate" turned out to be this trap. They are labelled <span class="chip illusion">An illusion</span> below.</p>
</section>

<section>
  <h2>Results by cancer type</h2>
  <p>Pick a cancer. Each card is one of the original questions, answered in plain language, with a label saying how much to trust it.</p>
  <div class="legend-chips">
    <span class="chip holds">Holds up</span><span>a real pattern in these patients</span>
    <span class="chip illusion">An illusion</span><span>an artefact of how survival was counted</span>
    <span class="chip unclear">No clear answer</span><span>too few patients or nothing beyond chance</span>
    <span class="chip descriptive">Description</span><span>a count or ranking, not a test</span>
  </div>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="panels"></div>
</section>

<section>
  <h2>What recurs across every cancer</h2>
  <ul class="recap">
    <li><strong>TP53 is the bad partner everywhere.</strong> In lung, breast, prostate and ALK-fusion tumours, adding a TP53 change to the main driver cut survival by a third to a half. It is the one gene that passed the statistical bar in dataset after dataset.</li>
    <li><strong>Losing the CDKN2A/B genes matters.</strong> It halved survival in bladder cancer and shortened it in pancreatic, ALK-fusion lung and glioblastoma patients.</li>
    <li><strong>Treatment comparisons need treatment dates.</strong> Every slide that compared treated with untreated patients from diagnosis overstated the benefit. Counting from the day the drug started removed it each time.</li>
    <li><strong>The DNA test itself can mislead.</strong> How often a gene change is found depends on which test the hospital used (the ERG rearrangement in prostate cancer ranges from 0% to 33% by test). Comparisons only count patients whose test could have found the change.</li>
  </ul>
</section>

<section>
  <h2>Glossary</h2>
  <details><summary>Mutation, deletion, amplification, fusion</summary><p>Four ways a gene can change in a tumour. A <em>mutation</em> is a spelling change in the gene. A <em>deep deletion</em> means both copies of the gene are gone. An <em>amplification</em> means many extra copies. A <em>fusion</em> is a rearrangement that joins two genes into one, often switching one of them permanently on (EML4-ALK in lung, TMPRSS2-ERG in prostate).</p></details>
  <details><summary>Median survival</summary><p>The time by which half the group had died. It is used instead of the average because many patients are still alive when the records stop, so the average cannot be computed.</p></details>
  <details><summary>Hazard ratio (HR)</summary><p>How much faster one group's risk of death runs than another's. HR 2 means twice the risk at any moment; HR 0.5 means half. On the technical page every comparison carries one, with a range in which the true value probably lies.</p></details>
  <details><summary>Why "statistical bar" and not just "bigger number"</summary><p>When you test hundreds of genes at once, some will look important by chance. The analysis corrects for this (a method called false-discovery-rate control) so that a gene labelled "holds up" is one unlikely to be a fluke.</p></details>
  <details><summary>Immortal time</summary><p>The time a patient must survive to qualify for a group, during which they could not have died while counted in that group. Counting it in their favour exaggerates any benefit. The fix is to start the clock at the moment of qualification (the treatment start or the biopsy).</p></details>
  <details><summary>Where the patients come from</summary><p>Memorial Sloan Kettering, Dana-Farber, Vanderbilt and Princess Margaret for the detailed cohorts; nineteen cancer centres in North America and Europe for the main GENIE database. All are academic hospitals, so the patients are not a cross-section of everyone with cancer.</p></details>
</section>
"""

TECH_BODY = """
<section class="hero compact">
  <p class="eyebrow">For Jacob · the 33 deck questions, replicated and extended</p>
  <h1>Survival by alteration across the GENIE and cBioPortal datasets</h1>
  <p class="lede">Each question is answered three ways: the deck's cBioPortal figure reproduced under its own definition, the corrected analysis next to it, and a one-line verdict. Charts are interactive; every number traces to a CSV in the repository and to the executed notebook for its dataset.</p>
  <div class="tiles" id="tiles"></div>
</section>

<section>
  <h2>Conventions used in every notebook</h2>
  <dl class="conv">
    <dt>Index date</dt><dd>Diagnosis of the index cancer (BPC <code>tt_os_dx</code>), as in the deck. Treatment questions: OS from regimen start (<code>tt_os_g</code>), metastatic-setting regimens only. AKT1: from metastatic diagnosis. GENIE v20: from the sequencing report (the only index available).</dd>
    <dt>Alterations</dt><dd><em>mut</em> = any non-synonymous somatic mutation (no OncoKB filter, the files lack it); <em>amp</em> = discrete +2; <em>homdel</em> = −2; <em>fus</em> = any structural variant on the gene. Questions that say "mutation" use <em>mut</em>; CDKN2A/B, MTAP, ERG and ALK questions use the type the deck names.</dd>
    <dt>One sample per patient</dt><dd>Earliest sequenced sample of the index cancer (BPC) or the primary (portal). A patient is wild-type for a gene only if the gene is on that sample's panel; otherwise excluded from that comparison.</dd>
    <dt>Profiles</dt><dd>Mutually exclusive groups for multi-gene questions; the overlap cBioPortal's group comparison drops (e.g. KEAP1+STK11, PIK3CA+TP53) is kept as its own group.</dd>
    <dt>Statistics</dt><dd>Kaplan–Meier with log-rank p (what cBioPortal shows) plus a Cox HR left-truncated at the sequencing report (<em>HR (LT)</em>); adjusted models add stage and age, subtype in breast, IDH in glioma, centre in v20. Screens: ≥10 per group, Benjamini–Hochberg FDR, q&lt;0.1 flagged.</dd>
    <dt>cBioPortal replication</dt><dd>Its "OS from start of treatment" attributes are per <em>regimen</em>; its group comparison excludes patients in both groups. Replicates use those exact definitions first.</dd>
  </dl>
</section>

<section>
  <h2>All verdicts</h2>
  <div class="filters" id="filters"></div>
  <div class="tablewrap"><table class="verdicts" id="vtable"><thead><tr><th>Dataset</th><th>Q</th><th>Verdict</th><th>Result</th></tr></thead><tbody></tbody></table></div>
</section>

<section>
  <h2>By dataset</h2>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="panels"></div>
</section>

<section>
  <h2>Open decisions</h2>
  <ul class="recap">
    <li><strong>N5 response definition.</strong> BPC PFS-I/M medians on osimertinib are ~5 months; time to next treatment (≥12 vs &lt;6 months) was used. If you prefer PFS-I, the screen is in <code>N5_pfs_i_screen_on_osimertinib.csv</code>.</li>
    <li><strong>Alteration scope.</strong> All non-synonymous variants counted. An OncoKB oncogenic filter would need the OncoKB API; it would mostly shrink the TP53 and KMT2D groups.</li>
    <li><strong>Treatment-effect questions (N4, A2, B2, Br2, Br4).</strong> Regimen-start OS with line-of-therapy tables is as far as these data go without a target-trial design; the pieces (regimen dates, lines, metastatic flags) are in the loader.</li>
    <li><strong>ERG in prostate.</strong> Any ERG-degrader framing should restrict to panels that call the rearrangement (MSK-IMPACT468, DFCI-ONCOPANEL-3, VICC-01-T7).</li>
    <li><strong>GENIE v20 small-n hits</strong> (EGFR, NF2, ROS1 mutations in ALK-fusion NSCLC, n≈13 each) are leads, not results.</li>
    <li><strong>DFCI metastatic breast cancer</strong> earned its place as an independent replication cohort (M1, M2); it could also serve C2-type pre/post questions (M3).</li>
  </ul>
</section>
"""

BOOT_LAY = "window.GENIE_PAGE='lay'; Site.init();"
BOOT_TECH = "window.GENIE_PAGE='tech'; Site.init();"


def page(title, desc, cls, body, boot, lay):
    head = HEAD.format(title=title, desc=desc, cls=cls, a1='aria-current="page"' if lay else "", a2='aria-current="page"' if not lay else "")
    return head + body + FOOT.format(repo=REPO, branch=BRANCH, boot=boot)


# ----------------------------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------------------------
if DIST.exists():
    shutil.rmtree(DIST)
(DIST / "figures").mkdir(parents=True)
for ds in DATASETS:
    src = RES / ds["id"]
    if src.exists():
        dst = DIST / "figures" / ds["id"]
        dst.mkdir()
        for p in src.glob("*.png"):
            shutil.copy(p, dst / p.name)
(DIST / "data.js").write_text("window.GENIE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
shutil.copy(ROOT / "site" / "site.css", DIST / "site.css")
shutil.copy(ROOT / "site" / "site.js", DIST / "site.js")
(DIST / "index.html").write_text(page("Cancer Genes and Survival", "Plain-language results of 32 questions about tumour DNA and survival, tested on AACR GENIE and cBioPortal data.", "lay", LAY_BODY, BOOT_LAY, True))
(DIST / "technical.html").write_text(page("GENIE Question Audit", "Replication and extension of 32 cBioPortal survival-by-alteration questions across ten GENIE and cBioPortal datasets.", "tech", TECH_BODY, BOOT_TECH, False))
n_fig = sum(1 for _ in (DIST / "figures").rglob("*.png"))
print(f"built {DIST}: curves {len(data['curves'])}, forests {len(data['forests'])}, figures {n_fig}, counts {counts}")
