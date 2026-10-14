"""Discover and download studies from the AACR GENIE cBioPortal instance via its REST API.

These studies are not on Synapse, only on genie.cbioportal.org (login required):
  - AACR Project GENIE AKT1 Cohort
  - DFCI-Profile Glioma Cohort 2013-2018 (DFCI, Nat Med 2020)
  - Metastatic Breast Cancer: 2013-2016 (DFCI, CCR 2020)

Prereqs:
  pip install requests pandas
  Log in at https://genie.cbioportal.org, open the user menu -> "Data Access Token", then
  export CBIO_TOKEN=<token>

Usage:
  python cbioportal_api_download.py --list                       # print all studies, flag likely matches
  python cbioportal_api_download.py --study <studyId> [--out data/genie_portal]
  python cbioportal_api_download.py --from-csv cbioportal_studies.csv   # every row with a study_id filled in

Manual fallback: the portal's Data Sets page offers a tar.gz per study for logged-in users.
"""
import argparse, hashlib, json, os, pathlib, sys
import pandas as pd
import requests

BASE = os.environ.get("CBIO_BASE", "https://genie.cbioportal.org/api")
TOKEN = os.environ.get("CBIO_TOKEN")
if not TOKEN:
    sys.exit("Set CBIO_TOKEN (Data Access Token from the genie.cbioportal.org user menu)")
S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"})

KEYWORDS = {"AKT1 cohort": ["akt1"], "DFCI glioma": ["glioma"], "DFCI metastatic breast": ["metastatic breast", "mbc"]}


def get(path, **params):
    r = S.get(f"{BASE}{path}", params=params, timeout=300)
    r.raise_for_status()
    return r.json()


def post(path, body, **params):
    r = S.post(f"{BASE}{path}", params=params, json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def list_studies():
    studies = get("/studies", projection="SUMMARY")
    df = pd.DataFrame(studies)[["studyId", "name", "allSampleCount", "citation", "pmid"]]
    print(df.to_string(index=False))
    print("\nLikely matches for the deck:")
    for label, kws in KEYWORDS.items():
        hits = df[df.name.str.lower().str.contains("|".join(kws))]
        print(f"  {label:24s} -> {', '.join(hits.studyId) or 'no match, check the list above'}")


def long_to_wide(rows, id_col):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index=id_col, columns="clinicalAttributeId", values="value", aggfunc="first").reset_index()


def download_study(study_id, out):
    dest = pathlib.Path(out) / study_id
    dest.mkdir(parents=True, exist_ok=True)
    written = []

    def save(df, name):
        p = dest / name
        df.to_csv(p, sep="\t", index=False)
        written.append({"study_id": study_id, "file": name, "rows": len(df),
                        "md5": hashlib.md5(p.read_bytes()).hexdigest()})
        print(f"  {name:40s} {len(df):>8,} rows")

    meta = get(f"/studies/{study_id}")
    (dest / "meta_study.json").write_text(json.dumps(meta, indent=2))

    # clinical attributes + patient/sample tables
    save(pd.DataFrame(get(f"/studies/{study_id}/clinical-attributes")), "clinical_attributes.tsv")
    save(long_to_wide(get(f"/studies/{study_id}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY"),
                      "patientId"), "data_clinical_patient.tsv")
    save(long_to_wide(get(f"/studies/{study_id}/clinical-data", clinicalDataType="SAMPLE", projection="SUMMARY"),
                      "sampleId"), "data_clinical_sample.tsv")

    # timeline (treatment, diagnosis, specimen, status events) - this is where OS-from-treatment-start comes from
    try:
        save(pd.json_normalize(get(f"/studies/{study_id}/clinical-events", projection="DETAILED")), "data_timeline.tsv")
    except requests.HTTPError as e:
        print(f"  (no clinical-events endpoint/data: {e.response.status_code})")

    # molecular profiles
    profiles = get(f"/studies/{study_id}/molecular-profiles")
    all_list = f"{study_id}_all"
    for prof in profiles:
        pid, alt = prof["molecularProfileId"], prof["molecularAlterationType"]
        if alt == "MUTATION_EXTENDED":
            rows = post(f"/molecular-profiles/{pid}/mutations/fetch", {"sampleListId": all_list}, projection="DETAILED")
            save(pd.json_normalize(rows), "data_mutations.tsv")
        elif alt == "COPY_NUMBER_ALTERATION" and prof["datatype"] == "DISCRETE":
            rows = post(f"/molecular-profiles/{pid}/discrete-copy-number/fetch", {"sampleListId": all_list},
                        discreteCopyNumberEventType="ALL", projection="DETAILED")
            save(pd.json_normalize(rows), "data_cna.tsv")
        elif alt == "STRUCTURAL_VARIANT":
            rows = post("/structural-variant/fetch", {"molecularProfileIds": [pid]})
            save(pd.json_normalize(rows), "data_sv.tsv")

    # gene panel per sample (which genes were tested)
    try:
        rows = post("/gene-panel-data/fetch", {"sampleMolecularIdentifiers": [
            {"molecularProfileId": p["molecularProfileId"], "sampleId": s["sampleId"]}
            for p in profiles if p["molecularAlterationType"] == "MUTATION_EXTENDED"
            for s in get(f"/studies/{study_id}/samples", projection="SUMMARY")]})
        save(pd.json_normalize(rows), "gene_panel_per_sample.tsv")
    except requests.HTTPError as e:
        print(f"  (gene-panel-data unavailable: {e.response.status_code})")

    log = pathlib.Path(out) / "download_log.csv"
    pd.DataFrame(written).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"-> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--study")
    ap.add_argument("--from-csv")
    ap.add_argument("--out", default="data/genie_portal")
    a = ap.parse_args()
    if a.list:
        list_studies()
    elif a.study:
        download_study(a.study, a.out)
    elif a.from_csv:
        for row in pd.read_csv(a.from_csv).dropna(subset=["study_id"]).itertuples():
            print(f"\n{row.deck_name} ({row.study_id})")
            download_study(row.study_id, a.out)
    else:
        ap.print_help()
