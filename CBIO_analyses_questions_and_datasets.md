# CBIO Analyses — Questions & Datasets

Oct 13, 2026 · @Daniel

## Purpose

Jacob's deck (*CBIO thoughts*, 31 slides) names 9 public cBioPortal datasets and 32 questions, almost all of one form: overall survival (OS) stratified by mutation or co-mutation profile within one cancer type, sometimes restricted to a treatment group. Every question below is numbered so it can be handed to a separate Claude Code run; the standard protocol in the fourth section applies to all of them.

The per-patient data Jacob wants for every analysis (slide 2):

- Tumor type and mutation profile, ideally pre-treatment and again after progression
- Treatment history
- Date of diagnosis
- Date of progression on first-line treatment
- Date of death or date of last follow-up

All figures Jacob cites are cBioPortal screenshots (Kaplan–Meier plots with log-rank p-values, oncoprints, CNA frequency tables). Each flagged *replicate* should be reproduced from downloaded data before any extension, as the validation step.

## Research questions

32 questions across 8 datasets, in Jacob's words with the comparison and endpoint made explicit. *Replicate* = a cBioPortal result already shown in the deck that must be reproduced first.

| ID | Dataset | Question (Jacob) | Groups compared | Endpoint | Slide |
| --- | --- | --- | --- | --- | --- |
| N1 | GENIE BPC NSCLC v2.0 | What is the OS of patients with EGFR mutations? | EGFR-mut vs EGFR-wt NSCLC | OS | 4 |
| N2 | GENIE BPC NSCLC v2.0 | How is OS in EGFR+ patients affected by co-occurring mutations? | Within EGFR-mut: each co-mutation present vs absent | OS | 4 |
| N3 | GENIE BPC NSCLC v2.0 | Which mutations most commonly co-occur with EGFR, and do any convey a negative prognosis? | Co-mutation frequency ranking; prognostic effect of each | Frequency; OS | 4 |
| N4 | GENIE BPC NSCLC v2.0 | *Replicate*: in EGFR+ patients osimertinib clearly outperforms chemotherapy | EGFR-mut on osimertinib vs on chemotherapy | OS from treatment start | 5 |
| N5 | GENIE BPC NSCLC v2.0 | Can we identify mutation profiles of EGFR+ patients who respond well vs poorly to osimertinib? | Within EGFR-mut osimertinib-treated: good vs poor responders (definition needed, see open questions) | Mutation profile differences | 5 |
| N6 | GENIE BPC NSCLC v2.0 | Compare EGFR vs KEAP1 vs STK11 vs KEAP1+STK11 — OS and response to different treatments | Four mutually defined groups; the KEAP1+STK11 overlap group that cBioPortal excluded must be mapped back in | OS; OS by treatment | 6–8 |
| A1 | GENIE AKT1 cohort | In patients receiving aromatase therapy, are specific mutations associated with worse OS? | Within AI-treated: each gene mut vs wt | OS | 9 |
| A2 | GENIE AKT1 cohort | *Replicate*: patients who received aromatase therapy lived longer than those who did not | AI-treated vs not | OS | 10 |
| A3 | GENIE AKT1 cohort | Do AI-treated patients with PIK3CA mutations do better or worse than those without? Ask this for each mutation | Within AI-treated: PIK3CA mut vs wt, then every gene | OS | 10 |
| A4 | GENIE AKT1 cohort | How do AI-treated patients perform with co-occurring PIK3CA + AKT1 + TP53 mutations? | Within AI-treated: triple-mut vs others; also pairwise combinations | OS | 10 |
| A5 | GENIE AKT1 cohort | 126 patients received CDK4/6 inhibitors, 123 concurrently with an AI. Survival curve for the combination, then which mutations predict poorer outcomes | Within AI + CDK4/6i-treated: each gene mut vs wt. Reference: [PMID 39754979](https://pubmed.ncbi.nlm.nih.gov/39754979/) | OS | 11 |
| G1 | DFCI-Profile Glioma 2013–2018 | *Replicate*: among glioblastoma, 135 patients who received temozolomide before resection had higher OS (log-rank p = 1.24e-8 in deck) | GBM: TMZ before resection yes vs no | OS | 12–13 |
| G2 | DFCI-Profile Glioma 2013–2018 | How does each CNA affect OS within each TMZ group? Do patients with CDKN2A/CDKN2B and MTAP deletions do worse? | GBM, stratified by TMZ: CDKN2A, CDKN2B, MTAP (HOMDEL), EGFR, CDK4, PDGFRA, MDM2, MDM4, PIK3C2B (AMP), PTEN (HOMDEL), GLI1 | OS | 13 |
| B1 | GENIE BPC Bladder v1.0 | How do bladder urothelial carcinoma patients do with CDKN2A or CDKN2B mutations? | Urothelial: CDKN2A/B altered vs not | OS | 14 |
| B2 | GENIE BPC Bladder v1.0 | How is OS affected by different treatments or combinations, especially in the CDKN2A/B group? | Urothelial: treatment regimens, stratified by CDKN2A/B status | OS from treatment start | 14 |
| Br1 | GENIE BPC BrCa v1.0 | *Replicate*: PIK3CA-mutant patients have significantly better OS than TP53-mutant patients | PIK3CA-mut vs TP53-mut | OS | 15–16 |
| Br2 | GENIE BPC BrCa v1.0 | Are there treatments that improved OS in TP53-mutant patients? | Within TP53-mut: by treatment regimen | OS from treatment start | 16 |
| Br3 | GENIE BPC BrCa v1.0 | Are there co-mutations in PIK3CA patients that caused worse outcomes? | Within PIK3CA-mut: each co-mutation present vs absent | OS | 16 |
| Br4 | GENIE BPC BrCa v1.0 | Break down OS from the start of different treatments | By regimen, from treatment start date | OS from treatment start | 16 |
| Br5 | GENIE BPC BrCa v1.0 | *Replicate*: PIK3CA-mut vs TP53-mut patients who received fulvestrant — no significant OS difference | Fulvestrant-treated: PIK3CA-mut vs TP53-mut | OS from treatment start | 17 |
| Br6 | GENIE BPC BrCa v1.0 | *Replicate*: on fulvestrant + palbociclib, PIK3CA-mut OS was better than TP53-mut | Fulvestrant+palbociclib-treated: PIK3CA-mut vs TP53-mut | OS from treatment start | 18 |
| C1 | GENIE BPC CRC v2.0 | *Replicate*: OS of patients biopsied before leucovorin vs after leucovorin was very different | Sample timing relative to leucovorin: pre vs post | OS | 19–21 |
| C2 | GENIE BPC CRC v2.0 | How do mutational profiles differ in samples taken before vs after chemotherapy? | Pre-treatment vs post-treatment samples (per regimen) | Mutation frequencies | 20 |
| C3 | GENIE BPC CRC v2.0 | How many patients have both pre and post biopsies, and how did their mutations change? | Paired samples within patient | Count; within-patient acquired/lost alterations | 20 |
| C4 | GENIE BPC CRC v2.0 | How does mutation status affect survival: KRAS vs TP53 vs APC, plus co-mutations (KRAS+TP53, KRAS+TP53+APC, …) | Single-gene groups and all co-mutation combinations | OS | 22–23 |
| P1 | GENIE BPC PANC v1.0 | How is OS of KRAS-mutant patients further differentiated by co-mutations? Which co-mutations associate with worse outcome? | Within KRAS-mut: each co-mutation present vs absent; CDKN2A/B and MTAP deletions of special interest | OS | 24 |
| Pr1 | GENIE BPC Prostate v1.0 | Characterize ERG alterations (context: Parabilis is developing an [ERG degrader](https://parabilismed.com/program/erg-degrader/)) | ERG-altered vs not | Frequency; OS | 25 |
| Pr2 | GENIE BPC Prostate v1.0 | *Replicate*: among ERG-altered patients, AR vs PTEN co-alteration shows no significant OS difference | Within ERG+: AR co-alt vs PTEN co-alt vs neither | OS | 26–27 |
| Pr3 | GENIE BPC Prostate v1.0 | Is there a way to sift the data for mutational profiles that convey higher risk in ERG+ patients? | Within ERG+: systematic screen of all co-alterations | OS | 27 |
| V1 | GENIE v20.0-public | Can OS be derived at all? Proposed: OS = (interval DOB→DOD or last contact) − (age at sequencing). Can diagnosis be inferred from age at sequencing? | Feasibility / endpoint derivation | Derived OS | 28, 30 |
| V2 | GENIE v20.0-public | 1,274 ALK-altered patients (predominantly lung): how does OS change with additional mutations? | Within ALK+: each co-mutation present vs absent | Derived OS | 29 |
| V3 | GENIE v20.0-public | Extract all mutations, structural variants and CNAs for every ALK+ patient (example: EML4-ALK with TP53, STAT3, EPHA3 mutations and ERBB2 amplification) and relate profiles to survival | Within ALK+: full alteration profile per patient | Derived OS | 31 |
| V4 | GENIE v20.0-public | No treatment data in v20; can treatments be identified another way? Reference: [Guardant study, ~300 patients](https://www.sciencedirect.com/science/article/pii/S1556086424006774) | Feasibility | — | 29 |

## Datasets

Nine datasets, all listed on slide 3 as cBioPortal studies. The six GENIE BPC releases carry curated treatment and outcome data; the three others carry sequencing plus limited clinical fields. Study IDs and access routes are marked *to confirm* where the deck does not state them.

| Dataset (as named in deck) | Focus in the deck | Treatment / outcome data | Size noted in deck | Questions |
| --- | --- | --- | --- | --- |
| GENIE BPC NSCLC v2.0-public | EGFR, KEAP1, STK11; osimertinib vs chemotherapy | Yes (BPC curated regimens, OS) | — | N1–N6 |
| AACR Project GENIE AKT1 Cohort | Aromatase inhibitors, CDK4/6 inhibitors; PIK3CA, AKT1, TP53 | Yes (AI and CDK4/6i exposure shown in deck) | 126 CDK4/6i-treated, 123 with concurrent AI | A1–A5 |
| DFCI-Profile Glioma Cohort 2013–2018 (DFCI, Nat Med 2020) | Glioblastoma; temozolomide before resection; CNAs | Partial (TMZ before resection flag, OS) | 135 TMZ-treated GBM; 926 CNA-profiled samples | G1–G2 |
| GENIE BPC BLADDER v1.0-public | Bladder urothelial carcinoma; CDKN2A/CDKN2B | Yes | — | B1–B2 |
| GENIE BPC BrCa v1.0-public | TP53 vs PIK3CA; fulvestrant, palbociclib | Yes | — | Br1–Br6 |
| GENIE BPC CRC v2.0-public | KRAS, TP53, APC; pre- vs post-chemotherapy biopsies (leucovorin) | Yes, with sample timing relative to treatment | — | C1–C4 |
| GENIE BPC PANC v1.0-public | KRAS co-mutations; CDKN2A/B and MTAP | Yes | — | P1 |
| GENIE BPC Prostate v1.0-public | ERG alterations; AR and PTEN co-alterations | Yes | — | Pr1–Pr3 |
| GENIE Cohort v20.0-public | ALK-altered patients (pan-cancer, mostly NSCLC) | No treatment data; OS must be derived from age at sequencing and DOB→DOD / last-contact intervals | 242,866 patients, 289,869 samples; 1,274 ALK-altered patients (1,153 samples) | V1–V4 |
| Metastatic Breast Cancer: 2013–2016 (DFCI, CCR 2020) | Listed on slide 3, no slides or questions | *to confirm* | — | none yet |

Access for reproducible runs (*to confirm* against current portal state):

- cBioPortal public studies (DFCI glioma, DFCI metastatic breast, GENIE AKT1): downloadable study archives from the cBioPortal datasets page, or the cBioPortal REST API; exact study IDs to be recorded in the analysis repo
- GENIE BPC public releases: hosted on the GENIE cBioPortal instance and on Synapse (AACR Project GENIE, registration and data-use terms required); Synapse gives the full clinical timeline tables (regimens, dates, PRISSMM outcomes) needed for OS-from-treatment-start analyses
- GENIE v20.0-public: Synapse release (registration required) or the GENIE cBioPortal instance; mutations, structural variants and CNA segments are separate files

## Standard analysis protocol

Every question above is a survival-by-alteration analysis, so one protocol can be run per question in parallel. Each Claude Code run should produce the same artifacts so results are comparable and the deck's cBioPortal figures can be checked first.

1. Pin the data: record dataset name, release version, download date, file checksums.
2. Define the cohort: cancer type filter (e.g. GBM only, urothelial only), one sample per patient (rule stated: earliest, or pre-treatment where C2/C3 need pairs), exclusions logged with counts (CONSORT-style table).
3. Define the alteration groups exactly: gene list, alteration types counted (SNV/indel, fusion, HOMDEL, AMP), oncogenic-only or any; mutually exclusive groups where the question compares profiles (N6, C4).
4. Define the index date and endpoint: OS from diagnosis, from sequencing, or from regimen start as the question states; death vs last contact as the event; note left truncation when follow-up starts at sequencing (all BPC and V1–V3).
5. Replicate the deck figure (rows marked *Replicate*) and report n, events, median OS, log-rank p alongside the deck's values.
6. Extend: Kaplan–Meier with risk tables; Cox models adjusted for age, sex, stage at diagnosis, and line of therapy where available; hazard ratios with 95% CI.
7. Control multiplicity for every "ask this for each mutation" screen (N2, N3, A3, Br3, P1, Pr3, V2): minimum group size, Benjamini–Hochberg FDR, pre-specified gene panel.
8. Flag treatment-comparison biases explicitly: immortal-time bias for treated-vs-untreated (A2, G1) and for pre- vs post-biopsy OS (C1); use landmark or time-varying exposure.
9. Output: one notebook or script per question, a results table (CSV), figures (SVG/PNG), and a short methods paragraph in the repo README; fixed random seeds and a pinned environment file.

## Open questions for Jacob

These decide cohort and endpoint definitions, so they should be settled before the parallel runs start.

- [ ] Data access: do we have Synapse credentials and GENIE data-use agreements, or should everything run from cBioPortal public downloads and the REST API?
- [ ] Priority: which 3–5 questions go first? Suggested starters are the six *Replicate* rows, since they validate the pipeline.
- [ ] Index date: for OS-by-mutation questions, measure from diagnosis, from sequencing, or from first-line start? The deck mixes all three.
- [ ] N5: how is "responds well" to osimertinib defined (time on treatment, progression within X months, OS threshold)?
- [ ] N6 and C4: should co-mutation groups be mutually exclusive (EGFR-only, KEAP1-only, KEAP1+STK11) or overlapping?
- [ ] Alteration scope: count all variants or only oncogenic/likely-oncogenic (OncoKB-annotated) ones? Include fusions and CNAs for every gene, or only where the deck names them (ERG, ALK, CDKN2A/B, MTAP)?
- [ ] V1: is the derived OS for GENIE v20 (DOB→DOD interval minus age at sequencing) acceptable given the left truncation, and should ALK+ mean fusions only or any ALK alteration?
- [ ] Metastatic Breast Cancer 2013–2016 (DFCI, CCR 2020): drop, or define questions?
- [ ] Covariates: which adjustments are expected for the publication target (age, sex, stage, histology, line of therapy, institution)? Any target journal or format?
- [ ] Gene panels for screens: the full BPC panel, or a curated list per cancer type?
