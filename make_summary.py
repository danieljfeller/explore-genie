"""Collect results/*/findings.csv into results/SUMMARY.md (one row per question) and print it."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
ORDER = ["bpc_nsclc", "genie_akt1", "dfci_glioma", "bpc_bladder", "bpc_brca", "bpc_crc", "bpc_panc", "bpc_prostate", "genie_v20", "dfci_mbc"]
NOTEBOOK = {"bpc_nsclc": "01_bpc_nsclc", "genie_akt1": "02_genie_akt1", "dfci_glioma": "03_dfci_glioma", "bpc_bladder": "04_bpc_bladder",
            "bpc_brca": "05_bpc_brca", "bpc_crc": "06_bpc_crc", "bpc_panc": "07_bpc_panc", "bpc_prostate": "08_bpc_prostate",
            "genie_v20": "09_genie_v20", "dfci_mbc": "10_dfci_mbc"}

rows, missing = [], []
for ds in ORDER:
    f = ROOT / "results" / ds / "findings.csv"
    if f.exists():
        df = pd.read_csv(f)
        df.insert(0, "dataset", ds)
        rows.append(df)
    else:
        missing.append(ds)
out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dataset", "question", "verdict", "note"])

lines = ["# Findings summary", "", "One line per question, copied from the last executed run of each notebook "
         "(`results/<dataset>/findings.csv`). Verdict vocabulary: *replicated* = the deck's cBioPortal figure reproduces under the deck's own "
         "definition; *signal* = something passed the pre-specified threshold (FDR < 0.1 for screens, p < 0.05 for single comparisons); "
         "*descriptive* = the question asked for a ranking, not a test.", ""]
if missing:
    lines += ["Not yet executed (data not downloaded): " + ", ".join(f"`{NOTEBOOK[m]}`" for m in missing), ""]
lines += ["| dataset | question | verdict | note |", "|---|---|---|---|"]
for r in out.itertuples():
    note = str(r.note).replace("|", "/")
    lines.append(f"| {r.dataset} | {r.question} | {r.verdict} | {note} |")
text = "\n".join(lines) + "\n"
(ROOT / "results" / "SUMMARY.md").write_text(text)
print(text)
