"""Download GENIE BPC + GENIE main-release files listed in genie_bpc_manifest.csv from Synapse.

Prereqs:
  pip install synapseclient pandas
  export SYNAPSE_AUTH_TOKEN=<personal access token>   # and accept the GENIE data-use terms on synapse.org

Usage:
  python download_genie_bpc.py [--cohorts NSCLC CRC GENIE_main] [--skip-optional] [--out data/genie]

Manifest categories: docs, clinical, analytic, timeline, genomic, panel, optional.
A 'panel_folder' row means: synapse_id is a folder (or any file inside it); download every child of that folder
whose name matches the filename pattern (used for the ~180 v20 gene-panel files).
"""
import argparse, fnmatch, hashlib, pathlib
import pandas as pd
import synapseclient

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="genie_bpc_manifest.csv")
ap.add_argument("--out", default="data/genie")
ap.add_argument("--cohorts", nargs="*", default=None)
ap.add_argument("--skip-optional", action="store_true")
args = ap.parse_args()

m = pd.read_csv(args.manifest)
if args.cohorts:
    m = m[m.cohort.isin(args.cohorts)]
if args.skip_optional:
    m = m[m.category != "optional"]

syn = synapseclient.login()  # reads SYNAPSE_AUTH_TOKEN
records = []

def fetch(cohort, category, syn_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    ent = syn.get(syn_id, downloadLocation=str(dest), ifcollision="overwrite.local")
    p = pathlib.Path(ent.path)
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    records.append(dict(cohort=cohort, category=category, filename=p.name, synapse_id=syn_id,
                        version=ent.versionNumber, size=p.stat().st_size, md5=md5, local_path=str(p)))
    print(f"{cohort:11s} {syn_id:12s} {p.name}")

for row in m.itertuples(index=False):
    dest = pathlib.Path(args.out) / row.cohort / row.category
    if row.category == "panel_folder":
        ent = syn.get(row.synapse_id, downloadFile=False)
        parent = row.synapse_id if "Folder" in ent.concreteType else ent.parentId   # a folder id, or any file inside the folder
        for child in syn.getChildren(parent, includeTypes=["file"]):
            if fnmatch.fnmatch(child["name"], row.filename):
                fetch(row.cohort, "panel", child["id"], pathlib.Path(args.out) / row.cohort / "panel")
    else:
        fetch(row.cohort, row.category, row.synapse_id, dest)

log = pathlib.Path(args.out) / "download_log.csv"
pd.DataFrame(records).to_csv(log, index=False)
print(f"\n{len(records)} files -> {log}  (commit this; gitignore the data itself)")
