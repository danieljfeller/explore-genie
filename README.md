# explore-genie

Survival-by-alteration analyses for the 32 questions in Jacob's *CBIO thoughts* deck, run on the public cBioPortal / AACR GENIE
datasets the deck names. The question list, with groups and endpoints made explicit, is in
[CBIO_analyses_questions_and_datasets.md](CBIO_analyses_questions_and_datasets.md). The one-line verdict per question is in
[results/SUMMARY.md](results/SUMMARY.md).

## Layout

| Path | What |
|---|---|
| `cbio_lib.py` | Shared code: data loaders (`BPC`, `Portal`, `GenieMain`), alteration matrices with panel-coverage masks, Kaplan–Meier / log-rank / Cox helpers, gene screens, plots. |
| `notebooks/NN_<dataset>.py` | One notebook per dataset, authored as a jupytext *percent* script (diffable). `notebooks/NN_<dataset>.ipynb` is the executed copy with outputs. |
| `results/<dataset>/` | One CSV and PNG per analysis, named `<question>_<slug>`, plus `findings.csv` (verdict per question). |
| `results/SUMMARY.md` | All findings in one table (`python make_summary.py`). |
| `download_genie_bpc.py`, `genie_bpc_manifest.csv` | Synapse download of the GENIE BPC cohorts and the GENIE v20 main release. |
| `cbioportal_api_download.py`, `cbioportal_studies.csv` | REST download of the three studies that exist only on genie.cbioportal.org. |
| `data/` | Downloaded data (git-ignored). `data/genie/<cohort>/` for Synapse files, `data/genie_portal/<study>/` for portal studies. Download logs with MD5s sit next to the data. |

| Notebook | Dataset | Questions | Status |
|---|---|---|---|
| `01_bpc_nsclc` | GENIE BPC NSCLC v2.0-public | N1–N6 | executed |
| `02_genie_akt1` | AACR GENIE AKT1 cohort | A1–A5 | executed |
| `03_dfci_glioma` | DFCI-Profile glioma 2013–2018 | G1–G2 | executed |
| `04_bpc_bladder` | GENIE BPC BLADDER v1.0-public | B1–B2 | written, needs download |
| `05_bpc_brca` | GENIE BPC BrCa v1.0-public | Br1–Br6 | executed |
| `06_bpc_crc` | GENIE BPC CRC v2.0-public | C1–C4 | written, needs download |
| `07_bpc_panc` | GENIE BPC PANC v1.0-public | P1 | written, needs download |
| `08_bpc_prostate` | GENIE BPC Prostate v1.0-public | Pr1–Pr3 | written, needs download |
| `09_genie_v20` | GENIE v20.0-public main release | V1–V4 | written, needs download |
| `10_dfci_mbc` | DFCI metastatic breast cancer 2013–2016 | none in deck; exploratory M1–M3 | executed |

## Running

Python 3.11 virtualenv `explore-genie` (pyenv, see `.python-version`). Packages: pandas, numpy, lifelines, statsmodels, scipy, matplotlib,
jupytext, jupyter, synapseclient, openpyxl.

```bash
# 1. Synapse data (needs a personal access token and acceptance of the GENIE data-use terms on synapse.org)
export SYNAPSE_AUTH_TOKEN=...
python download_genie_bpc.py --cohorts CRC PANC Prostate BLADDER GENIE_main

# 2. genie.cbioportal.org studies (needs the portal's Data Access Token)
export CBIO_TOKEN=...
python cbioportal_api_download.py --from-csv cbioportal_studies.csv

# 3. Execute a notebook (writes the .ipynb next to the .py and the tables/figures under results/)
cd notebooks && jupytext --to ipynb --execute 06_bpc_crc.py

# 4. Refresh the findings table
python make_summary.py
```

## Analysis conventions (the defaults chosen for the first pass)

These are the decisions the deck left open. They are applied identically in every notebook and stated again at the top of each one.

- **Index date:** diagnosis of the index cancer (BPC `tt_os_dx`), as the deck's cBioPortal figures. Treatment questions use OS from regimen
  start (`tt_os_g`), restricted to regimens started at or after the first distant metastasis so adjuvant and metastatic therapy are not mixed.
  The AKT1 cohort uses OS from metastatic diagnosis; the GENIE v20 release only allows OS from the sequencing report.
- **Alteration definitions:** *mut* = any non-synonymous somatic mutation (missense, nonsense, frameshift, splice site, in-frame indel,
  nonstop, start-site). No OncoKB filtering (the Synapse and REST files do not carry the annotation). *amp* = discrete copy number +2,
  *homdel* = −2, *fus* = any structural variant involving the gene. *alt* = any of these, which is cBioPortal's default query semantics.
  Questions that say "mutation" use *mut*; CDKN2A/B, MTAP, ERG and ALK questions use the alteration type the deck names.
- **One sample per patient:** the earliest sequenced sample of the index cancer (BPC) or the primary sample (portal studies).
  A patient counts as wild-type for a gene only if that gene is on the sample's panel; otherwise the patient is left out of that comparison.
- **Mutually exclusive profiles** for the multi-gene questions (N6, C4, A4, Br1): e.g. *EGFR only*, *KEAP1 only*, *KEAP1+STK11*, *none*.
  The overlap group that cBioPortal's group comparison drops is kept as its own group.
- **Statistics:** Kaplan–Meier with log-rank p (what cBioPortal shows), plus a Cox hazard ratio **left-truncated at the sequencing report**
  (patients enter GENIE only when sequenced, so OS from diagnosis is otherwise immortal-time biased). Adjusted models add stage and age
  (and subtype in breast, IDH in glioma). Screens over many genes require ≥ 10 patients in each group and report Benjamini–Hochberg FDR;
  treat q < 0.1 as "worth a look".
- **cBioPortal replication:** its "OS from start of treatment" attributes are defined **per regimen** (e.g. `Fulvestrant` vs
  `Fulvestrant, Palbociclib`), and its group comparison excludes patients in both groups. Replicates use those exact definitions first,
  then show the broader definition.
- **Time to next treatment** (`ttnt_ca_seq`, next regimen or death) is used as the response proxy for N5, because BPC progression-free
  survival counts any imaging or oncologist note of progression and its medians are much shorter than trial PFS.

## Known limitations

- Immortal-time bias dominates several deck figures (G1, C1, A2, N4 from diagnosis): patients had to survive to receive a treatment or a
  second biopsy. Each notebook shows the naive replication and the corrected version side by side.
- The AKT1 cohort is a case-control design around AKT1 mutations; 25% of its samples come from panels whose gene lists are not in the GENIE
  panel files and are assumed to cover every gene.
- Structural-variant calling differs by panel, so fusion frequencies (ERG, ALK) partly reflect which panels detect them.
- No multiple-testing correction is applied across questions, only within each screen.
