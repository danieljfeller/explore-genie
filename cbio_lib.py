"""Shared loaders and survival helpers for the CBIO / GENIE question set.

Conventions used by every notebook (see README.md):

* time is in MONTHS; event = 1 for death, 0 for censored
* "mut"    = non-synonymous somatic mutation (classes in NONSYN)
* "amp"    = discrete copy-number +2, "homdel" = discrete copy-number -2
* "fus"    = structural variant / fusion that involves the gene
* "alt"    = any of the above (cBioPortal's default query semantics)
* a patient counts as "tested" for a gene only if the gene is on the panel of the
  sample used for that patient; untested patients are dropped from that gene's
  comparison instead of being called wild-type
* one sample per patient = earliest sequenced sample of the index cancer (BPC) or the
  primary sample (portal studies); notebooks override this where a question needs it
"""
from __future__ import annotations

import glob
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

NONSYN = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation", "Splice_Site",
    "Frame_Shift_Del", "Frame_Shift_Ins", "Frame_Shift_DEL", "Frame_Shift_INS",
    "In_Frame_Del", "In_Frame_Ins", "In_Frame_DEL", "In_Frame_INS", "Translation_Start_Site",
}

# drug classes used to build treatment groups from BPC regimen strings
EGFR_TKI = ["Osimertinib", "Erlotinib", "Gefitinib", "Afatinib", "Dacomitinib"]
ALK_TKI = ["Crizotinib", "Alectinib", "Brigatinib", "Lorlatinib", "Ceritinib"]
IO = ["Nivolumab", "Pembrolizumab", "Atezolizumab", "Durvalumab", "Ipilimumab", "Cemiplimab"]
CHEMO = ["Carboplatin", "Cisplatin", "Oxaliplatin", "Pemetrexed", "Paclitaxel", "Nabpaclitaxel", "Docetaxel",
         "Gemcitabine", "Vinorelbine", "Etoposide", "Capecitabine", "Fluorouracil", "Irinotecan", "Leucovorin",
         "Cyclophosphamide", "Doxorubicin", "Eribulin", "Methotrexate", "Mitomycin", "Vinblastine", "Ixabepilone"]
AROMATASE_INHIBITORS = ["Letrozole", "Anastrozole", "Exemestane"]
CDK46 = ["Palbociclib", "Ribociclib", "Abemaciclib"]

# validated categorical palette (dataviz reference instance, light surface)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def setup_plots():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2,
    })


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def contains_any(series: pd.Series, drugs) -> pd.Series:
    """Case-insensitive 'any of these drug names appears in the regimen string'."""
    pat = "|".join(re.escape(d) for d in drugs)
    return series.fillna("").str.contains(pat, case=False, regex=True)


def consort(steps):
    """steps: list of (label, n). Returns a CONSORT-style table with the drop at each step."""
    df = pd.DataFrame(steps, columns=["step", "n"])
    df["dropped"] = (-df.n.diff()).fillna(0).astype(int)
    return df


# --------------------------------------------------------------------------------------
# Gene panels
# --------------------------------------------------------------------------------------

def load_panel_genes() -> dict[str, set]:
    """stable_id -> gene set, from every data_gene_panel_*.txt under data/."""
    panels = {}
    for f in glob.glob(str(DATA / "**" / "data_gene_panel_*.txt"), recursive=True):
        sid, genes = None, set()
        for line in open(f):
            if line.startswith("stable_id:"):
                sid = line.split(":", 1)[1].strip()
            elif line.startswith("gene_list:"):
                genes = set(line.split(":", 1)[1].split())
        if sid and genes:
            panels[sid] = genes
    return panels


# --------------------------------------------------------------------------------------
# Alteration matrices
# --------------------------------------------------------------------------------------

@dataclass
class Alterations:
    """Patient x gene boolean matrices plus 'tested' masks."""
    mut: pd.DataFrame
    amp: pd.DataFrame
    homdel: pd.DataFrame
    fus: pd.DataFrame
    tested: pd.DataFrame       # mutation / fusion panel coverage
    cna_tested: pd.DataFrame   # CNA profiled
    sample_of: pd.Series       # patient -> sample id(s) used
    n_unknown_panel: int = 0

    KINDS = ("mut", "amp", "homdel", "fus")

    @property
    def patients(self):
        return self.mut.index

    @property
    def genes(self):
        return self.mut.columns

    def matrix(self, kind):
        if kind == "alt":
            return self.mut | self.amp | self.homdel | self.fus
        if kind == "cna":
            return self.amp | self.homdel
        return getattr(self, kind)

    def tested_for(self, kind):
        if kind in ("amp", "homdel", "cna"):
            return self.cna_tested
        if kind == "alt":
            return self.tested | self.cna_tested
        return self.tested

    def status(self, gene, kinds=("mut",)) -> pd.Series:
        """Boolean altered-status per patient, restricted to patients tested for `gene`."""
        kinds = self._kinds(kinds)
        if gene not in self.genes:
            raise KeyError(f"{gene} not in alteration matrix")
        tested = pd.Series(False, index=self.patients)
        altered = pd.Series(False, index=self.patients)
        for k in kinds:
            t = self.tested_for(k)[gene]
            tested |= t
            altered |= self.matrix(k)[gene] & t
        return altered[tested]

    def any_of(self, genes, kinds=("mut",)) -> pd.Series:
        """Altered in any of `genes`; restricted to patients tested for all of them."""
        cols = [self.status(g, kinds) for g in genes]
        df = pd.concat(cols, axis=1, join="inner")
        return df.any(axis=1)

    def combo_groups(self, genes, kinds=("mut",), min_n=10, wt_label="WT", other_label="other combination") -> pd.Series:
        """Mutually exclusive profile labels like 'EGFR', 'KEAP1+STK11', 'WT'.
        Restricted to patients tested for every gene. Rare combinations (< min_n) -> other_label."""
        cols = pd.concat([self.status(g, kinds).rename(g) for g in genes], axis=1, join="inner")
        lab = cols.apply(lambda r: "+".join([g for g in genes if r[g]]) or wt_label, axis=1)
        counts = lab.value_counts()
        rare = counts[counts < min_n].index
        return lab.where(~lab.isin(rare), other_label).rename("profile")

    def frequency(self, kinds=("mut",), patients=None, genes=None, min_n=1) -> pd.DataFrame:
        """Alteration frequency per gene among `patients` (default all), per kind."""
        kinds = self._kinds(kinds)
        rows = []
        for k in kinds:
            M, T = self.matrix(k), self.tested_for(k)
            if patients is not None:
                M, T = M.loc[M.index.intersection(patients)], T.loc[T.index.intersection(patients)]
            if genes is not None:
                M, T = M[[g for g in genes if g in M.columns]], T[[g for g in genes if g in T.columns]]
            n_alt = (M & T).sum()
            n_tested = T.sum()
            df = pd.DataFrame({"gene": n_alt.index, "kind": k, "n_altered": n_alt.values, "n_tested": n_tested.values})
            rows.append(df[df.n_altered >= min_n])
        out = pd.concat(rows, ignore_index=True)
        out["pct"] = (100 * out.n_altered / out.n_tested).round(1)
        return out.sort_values("n_altered", ascending=False).reset_index(drop=True)

    def cooccurrence(self, anchor: pd.Series, kinds=("mut",), min_n=5, genes=None) -> pd.DataFrame:
        """Enrichment of each gene feature in anchor-positive vs anchor-negative patients.
        anchor: boolean Series over patients (e.g. alts.status('EGFR'))."""
        kinds = self._kinds(kinds)
        rows = []
        for k in kinds:
            M, T = self.matrix(k), self.tested_for(k)
            for g in (genes or M.columns):
                if g not in M.columns:
                    continue
                idx = anchor.index.intersection(T.index[T[g]])
                if len(idx) == 0:
                    continue
                a = anchor.loc[idx].astype(bool)
                s = M.loc[idx, g]
                n_pos, n_neg = int(a.sum()), int((~a).sum())
                alt_pos, alt_neg = int((s & a).sum()), int((s & ~a).sum())
                if alt_pos + alt_neg < min_n:
                    continue
                _, p = fisher_exact([[alt_pos, n_pos - alt_pos], [alt_neg, n_neg - alt_neg]])
                rows.append(dict(gene=g, kind=k, n_anchor=n_pos, altered_in_anchor=alt_pos,
                                 pct_in_anchor=round(100 * alt_pos / n_pos, 1) if n_pos else np.nan,
                                 n_rest=n_neg, altered_in_rest=alt_neg,
                                 pct_in_rest=round(100 * alt_neg / n_neg, 1) if n_neg else np.nan, p_fisher=p))
        out = pd.DataFrame(rows)
        if len(out):
            out["q_bh"] = multipletests(out.p_fisher, method="fdr_bh")[1]
            out = out.sort_values("altered_in_anchor", ascending=False).reset_index(drop=True)
        return out

    def _kinds(self, kinds):
        if isinstance(kinds, str):
            kinds = (kinds,)
        if "alt" in kinds:
            return self.KINDS
        return tuple(kinds)


def build_alterations(samples: pd.DataFrame, mut_long: pd.DataFrame, cna_long: pd.DataFrame, sv_long: pd.DataFrame,
                      panel_of: pd.Series, panel_genes: dict, genes=None) -> Alterations:
    """samples: DataFrame[sample_id, patient_id]. Several samples per patient -> union of alterations.
    mut_long: [sample_id, gene] (already non-synonymous). cna_long: [sample_id, gene, value] for every
    profiled gene (value 0 included). sv_long: [sample_id, gene]. panel_of: sample_id -> panel stable_id."""
    samples = samples[["sample_id", "patient_id"]].drop_duplicates().reset_index(drop=True)
    sid2pid = samples.set_index("sample_id")["patient_id"]
    patients = pd.Index(samples.patient_id.unique())
    sample_panels = panel_of.reindex(samples.sample_id)
    known = [p for p in sample_panels.dropna().unique() if p in panel_genes]
    if genes is None:
        g = set(mut_long.gene) | set(cna_long.gene) | set(sv_long.gene)
        for p in known:
            g |= panel_genes[p]
        genes = sorted(x for x in g if isinstance(x, str))
    genes = pd.Index(genes)

    def wide(long):
        if long is None or long.empty:
            return pd.DataFrame(False, index=patients, columns=genes)
        sub = long[long.sample_id.isin(sid2pid.index) & long.gene.isin(genes)]
        if sub.empty:
            return pd.DataFrame(False, index=patients, columns=genes)
        ct = pd.crosstab(sub.sample_id.map(sid2pid), sub.gene) > 0
        return ct.reindex(index=patients, columns=genes, fill_value=False)

    mut = wide(mut_long)
    amp = wide(cna_long[cna_long.value == 2])
    homdel = wide(cna_long[cna_long.value == -2])
    fus = wide(sv_long)
    cna_tested = wide(cna_long)

    # mutation-panel coverage, vectorised: panels x genes, then samples -> patients
    panel_matrix = pd.DataFrame({p: [g in panel_genes[p] for g in genes] for p in known}, index=genes).T
    tested_s = panel_matrix.reindex(sample_panels.values)
    n_unknown = int(tested_s.isna().all(axis=1).sum())
    tested_s = tested_s.fillna(True).astype(bool)
    tested_s.index = samples.sample_id.values
    tested = tested_s.groupby(samples.patient_id.values).any().reindex(patients).fillna(True).astype(bool)
    tested = tested.reindex(columns=genes).fillna(False)

    sample_of = samples.groupby("patient_id").sample_id.agg(lambda s: ";".join(s)).reindex(patients)
    return Alterations(mut, amp, homdel, fus, tested, cna_tested, sample_of, n_unknown)


# --------------------------------------------------------------------------------------
# GENIE BPC loader (Synapse analytic + genomic files)
# --------------------------------------------------------------------------------------

class BPC:
    """One GENIE BPC cohort (NSCLC, BrCa, CRC, PANC, Prostate, BLADDER)."""

    def __init__(self, cohort: str, data_dir: Path = DATA / "genie"):
        self.cohort = cohort
        d = self.dir = Path(data_dir) / cohort
        self.patients = pd.read_csv(d / "analytic/patient_level_dataset.csv", low_memory=False)
        ci = pd.read_csv(d / "analytic/cancer_level_dataset_index.csv", low_memory=False)
        ci = ci.sort_values(["record_id", "dob_ca_dx_days"])
        self.n_index_cancers = len(ci)
        self.n_multi_index = int((ci.groupby("record_id").size() > 1).sum())
        self.index = ci.drop_duplicates("record_id", keep="first").set_index("record_id")
        # months from diagnosis to first distant metastasis (0 if stage IV at diagnosis, NaN if never)
        ix = self.index
        met = pd.Series(np.nan, index=ix.index)
        if "dx_to_dmets_mos" in ix.columns:
            met = ix.dx_to_dmets_mos.astype(float)
        if "stage_dx" in ix.columns:
            met = met.where(ix.stage_dx != "Stage IV", 0.0)
        self.index["met_time_mos"] = met

        rg = pd.read_csv(d / "analytic/regimen_cancer_level_dataset.csv", low_memory=False)
        key = self.index.reset_index()[["record_id", "ca_seq"]]
        self.regimens = rg.merge(key, on=["record_id", "ca_seq"])

        cpt = pd.read_csv(d / "analytic/cancer_panel_test_level_dataset.csv", low_memory=False)
        cpt = cpt.merge(key, on=["record_id", "ca_seq"]).sort_values(["record_id", "dx_cpt_rep_days", "cpt_number"])
        cpt = cpt.rename(columns={"cpt_genie_sample_id": "sample_id", "record_id": "patient_id"})
        self.samples_all = cpt
        self.samples = cpt.drop_duplicates("patient_id", keep="first").set_index("patient_id", drop=False)

        maf = pd.read_csv(d / "genomic/data_mutations_extended.txt", sep="\t", low_memory=False,
                          usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short"])
        maf = maf.rename(columns={"Hugo_Symbol": "gene", "Tumor_Sample_Barcode": "sample_id"})
        self.maf = maf
        self.mut_long = maf[maf.Variant_Classification.isin(NONSYN)].dropna(subset=["gene"])

        cna = pd.read_csv(d / "genomic/data_CNA.txt", sep="\t", index_col=0)
        self.cna_wide = cna
        cl = cna.stack(future_stack=True).dropna().rename("value").reset_index()
        cl.columns = ["gene", "sample_id", "value"]
        self.cna_long = cl

        if (d / "genomic/data_sv.txt").exists():
            sv = pd.read_csv(d / "genomic/data_sv.txt", sep="\t", low_memory=False)
            self.sv_raw = sv
            long = pd.concat([sv[["Sample_Id", "Site1_Hugo_Symbol", "Event_Info"]].rename(columns={"Site1_Hugo_Symbol": "gene"}),
                              sv[["Sample_Id", "Site2_Hugo_Symbol", "Event_Info"]].rename(columns={"Site2_Hugo_Symbol": "gene"})])
            self.sv_long = long.rename(columns={"Sample_Id": "sample_id", "Event_Info": "detail"}).dropna(subset=["gene"]).drop_duplicates()
        elif (d / "genomic/data_fusions.txt").exists():
            fu = pd.read_csv(d / "genomic/data_fusions.txt", sep="\t", low_memory=False)
            self.sv_raw = fu
            self.sv_long = fu.rename(columns={"Tumor_Sample_Barcode": "sample_id", "Hugo_Symbol": "gene", "Fusion": "detail"})[
                ["sample_id", "gene", "detail"]].dropna(subset=["gene"]).drop_duplicates()
        else:
            self.sv_raw = self.sv_long = pd.DataFrame(columns=["sample_id", "gene", "detail"])

        gm = pd.read_csv(d / "panel/data_gene_matrix.txt", sep="\t")
        self.panel_of = gm.set_index("SAMPLE_ID")["mutations"]
        self.panel_genes = load_panel_genes()

    # ---- cohorts and endpoints -------------------------------------------------------
    def alterations(self, genes=None, rule="earliest") -> Alterations:
        s = self.samples if rule == "earliest" else self.samples_all
        return build_alterations(s[["sample_id", "patient_id"]], self.mut_long, self.cna_long, self.sv_long,
                                 self.panel_of, self.panel_genes, genes)

    def survival_dx(self) -> pd.DataFrame:
        """OS from diagnosis of the index cancer (months). entry = months from diagnosis to the sequencing
        report of the sample used (left-truncation time), when available."""
        ix = self.index
        cols = {c: c for c in ["age_dx", "stage_dx", "ca_hist_adeno_squamous", "ca_histology", "naaccr_histology_cd", "bca_subtype",
                               "institution", "ca_type", "dob_ca_dx_yrs"] if c in ix.columns}
        out = pd.DataFrame({"time": ix.tt_os_dx_mos, "event": ix.os_dx_status.astype(int)})
        for c in cols:
            out[c] = ix[c]
        sex = self.patients.set_index("record_id").naaccr_sex_code
        out["sex"] = sex.reindex(out.index)
        out["entry"] = self.samples.dx_cpt_rep_mos.reindex(out.index)
        out["sample_type"] = self.samples.sample_type.reindex(out.index)
        out.index.name = "patient_id"
        return out

    def survival_regimen(self, pattern=None, exact=None, exclude=None, which="first", line=None) -> pd.DataFrame:
        """OS from the start of the first regimen (per patient) matching the drug pattern.
        pattern / exclude: case-insensitive regex on the regimen string; exact: regimen string equality;
        line: restrict to regimen_number_within_cancer."""
        rg = self.regimens
        m = pd.Series(True, index=rg.index)
        if pattern is not None:
            m &= rg.regimen_drugs.fillna("").str.contains(pattern, case=False, regex=True)
        if exact is not None:
            m &= rg.regimen_drugs == exact
        if exclude is not None:
            m &= ~rg.regimen_drugs.fillna("").str.contains(exclude, case=False, regex=True)
        if line is not None:
            m &= rg.regimen_number_within_cancer == line
        sub = rg[m].sort_values(["record_id", "dx_reg_start_int"])
        sub = sub.drop_duplicates("record_id", keep="first" if which == "first" else "last").set_index("record_id")
        out = pd.DataFrame({
            "regimen": sub.regimen_drugs, "line": sub.regimen_number_within_cancer,
            "start_mos": sub.dx_reg_start_int_mos, "time": sub.tt_os_g_mos, "event": sub.os_g_status.astype(int),
            "clinical_trial": sub.drugs_ct_yn,
        })
        for src, dst in [("tt_pfs_i_or_m_g_mos", "pfs_time"), ("pfs_i_or_m_g_status", "pfs_event"),
                         ("tt_pfs_i_g_mos", "pfs_i_time"), ("pfs_i_g_status", "pfs_i_event"),
                         ("ttnt_ca_seq_mos", "ttnt_time"), ("ttnt_ca_seq_status", "ttnt_event")]:
            if src in sub.columns:
                out[dst] = sub[src]
        seq = self.samples.dx_cpt_rep_mos.reindex(out.index)
        out["entry"] = (seq - out.start_mos).clip(lower=0)
        met = self.index.met_time_mos.reindex(out.index)
        out["met_setting"] = (out.start_mos >= met).fillna(False)   # regimen started at/after first distant metastasis
        out.index.name = "patient_id"
        return out

    def summary(self):
        return {
            "cohort": self.cohort, "patients": len(self.patients), "index_cancers": self.n_index_cancers,
            "patients_with_2+_index_cancers": self.n_multi_index, "samples": len(self.samples_all),
            "samples_used": len(self.samples), "regimens": len(self.regimens),
            "mutations_nonsyn": len(self.mut_long), "cna_profiled_samples": self.cna_wide.shape[1],
            "sv_rows": len(self.sv_long),
        }


# --------------------------------------------------------------------------------------
# cBioPortal (REST download) loader
# --------------------------------------------------------------------------------------

class Portal:
    """One study downloaded with cbioportal_api_download.py (AKT1, DFCI glioma, DFCI MBC)."""

    def __init__(self, study: str, data_dir: Path = DATA / "genie_portal"):
        self.study = study
        d = self.dir = Path(data_dir) / study
        self.patients = pd.read_csv(d / "data_clinical_patient.tsv", sep="\t", low_memory=False).set_index("patientId")
        sm = pd.read_csv(d / "data_clinical_sample.tsv", sep="\t", low_memory=False)
        sm["patient_id"] = sm.sampleId.str.extract(r"^(GENIE-[A-Z]+-[^-]+(?:-P-\d+)?)")[0]
        # robust patient id: use the mutation/CNA tables' patientId mapping when available
        mut = pd.read_csv(d / "data_mutations.tsv", sep="\t", low_memory=False)
        cna = pd.read_csv(d / "data_cna.tsv", sep="\t", low_memory=False)
        pid_map = pd.concat([mut[["sampleId", "patientId"]], cna[["sampleId", "patientId"]]]).drop_duplicates().set_index("sampleId").patientId
        gp = pd.read_csv(d / "gene_panel_per_sample.tsv", sep="\t")
        pid_map = pd.concat([pid_map, gp.set_index("sampleId").patientId]).groupby(level=0).first()
        sm["patient_id"] = sm.sampleId.map(pid_map).fillna(sm.patient_id)
        self.samples_all = sm.rename(columns={"sampleId": "sample_id"})
        self.samples = self.choose_samples()

        self.mut_raw = mut
        self.mut_long = (mut[mut.mutationType.isin(NONSYN)]
                         .rename(columns={"sampleId": "sample_id", "gene.hugoGeneSymbol": "gene", "proteinChange": "HGVSp_Short"})
                         [["sample_id", "gene", "HGVSp_Short", "mutationType"]].dropna(subset=["gene"]))
        self.cna_long = cna.rename(columns={"sampleId": "sample_id", "gene.hugoGeneSymbol": "gene", "alteration": "value"})[
            ["sample_id", "gene", "value"]]
        svp = d / "data_sv.tsv"
        if svp.exists() and svp.stat().st_size > 0:
            sv = pd.read_csv(svp, sep="\t", low_memory=False)
            self.sv_raw = sv
            parts = []
            for c in ["site1HugoSymbol", "site2HugoSymbol"]:
                if c in sv.columns:
                    parts.append(sv[["sampleId", c, "eventInfo"]].rename(columns={c: "gene", "eventInfo": "detail", "sampleId": "sample_id"}))
            self.sv_long = pd.concat(parts).dropna(subset=["gene"]).drop_duplicates() if parts else pd.DataFrame(columns=["sample_id", "gene", "detail"])
        else:
            self.sv_raw = self.sv_long = pd.DataFrame(columns=["sample_id", "gene", "detail"])
        self.panel_of = gp.drop_duplicates("sampleId").set_index("sampleId").genePanelId
        self.panel_genes = load_panel_genes()

    def choose_samples(self, prefer_primary=True) -> pd.DataFrame:
        sm = self.samples_all.copy()
        if prefer_primary and "SAMPLE_TYPE" in sm.columns:
            sm["_rank"] = (~sm.SAMPLE_TYPE.astype(str).str.lower().str.startswith("prim")).astype(int)
            sm = sm.sort_values(["patient_id", "_rank", "sample_id"]).drop(columns="_rank")
        else:
            sm = sm.sort_values(["patient_id", "sample_id"])
        return sm.drop_duplicates("patient_id").set_index("patient_id", drop=False)

    def alterations(self, genes=None, samples=None) -> Alterations:
        s = self.samples if samples is None else samples
        return build_alterations(s[["sample_id", "patient_id"]], self.mut_long, self.cna_long, self.sv_long,
                                 self.panel_of, self.panel_genes, genes)

    def survival(self, time_col="OS_MONTHS", status_col="OS_STATUS") -> pd.DataFrame:
        p = self.patients
        out = p.copy()
        out["time"] = pd.to_numeric(p[time_col], errors="coerce")
        out["event"] = p[status_col].astype(str).str.startswith("1").astype(int)
        out.loc[p[status_col].isna(), "event"] = np.nan
        out.index.name = "patient_id"
        return out


# --------------------------------------------------------------------------------------
# Survival statistics and plots
# --------------------------------------------------------------------------------------

def _clean(df, group, time, event):
    d = df.dropna(subset=[group, time, event]).copy()
    d[event] = d[event].astype(int)
    return d


def logrank_p(df, group, time="time", event="event"):
    d = _clean(df, group, time, event)
    levels = d[group].unique()
    if len(levels) < 2:
        return np.nan
    if len(levels) == 2:
        a, b = d[d[group] == levels[0]], d[d[group] == levels[1]]
        return logrank_test(a[time], b[time], a[event], b[event]).p_value
    return multivariate_logrank_test(d[time], d[group], d[event]).p_value


def km_table(df, group, time="time", event="event", order=None) -> pd.DataFrame:
    """n, events, median OS and 1/2-year survival per group; log-rank p in .attrs['p_logrank']."""
    d = _clean(df, group, time, event)
    order = list(order) if order is not None else sorted(d[group].unique(), key=str)
    rows = []
    for g in order:
        sub = d[d[group] == g]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter().fit(sub[time], sub[event])
        s = kmf.survival_function_at_times([12, 24, 60]).values
        rows.append(dict(group=g, n=len(sub), events=int(sub[event].sum()),
                         median_months=round(kmf.median_survival_time_, 1),
                         surv_1y=round(s[0], 3), surv_2y=round(s[1], 3), surv_5y=round(s[2], 3)))
    out = pd.DataFrame(rows)
    p = logrank_p(d, group, time, event)
    out["p_logrank"] = p
    out.attrs["p_logrank"] = p
    return out


def km_plot(df, group, time="time", event="event", title="", order=None, ax=None, xmax=None, risk_table=True,
            xlabel="Months", colors=None, legend_loc="upper right", subtitle=None):
    """Kaplan-Meier curves per group with censor ticks, legend with n/events/median, log-rank p, at-risk table."""
    d = _clean(df, group, time, event)
    order = [g for g in (order if order is not None else sorted(d[group].unique(), key=str)) if (d[group] == g).any()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = colors or PALETTE
    fitters = []
    for color, g in zip(colors, order):
        sub = d[d[group] == g]
        kmf = KaplanMeierFitter(label=str(g)).fit(sub[time], sub[event])
        med = kmf.median_survival_time_
        label = f"{g}  (n={len(sub)}, events={int(sub[event].sum())}, median={'NR' if np.isinf(med) else f'{med:.1f}'})"
        kmf._label = label
        kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=color, linewidth=2,
                                   censor_styles={"ms": 5, "marker": "|", "alpha": 0.6})
        fitters.append(kmf)
    p = logrank_p(d, group, time, event)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, xmax or min(d[time].max() * 1.02, d[time].quantile(0.99) * 1.1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Overall survival")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.text(0.98, 0.95, f"log-rank p = {fmt_p(p)}", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=9)
    ax.legend(loc=legend_loc, bbox_to_anchor=(0.98, 0.9) if legend_loc == "upper right" else None)
    if risk_table and len(order) <= 4:
        add_at_risk_counts(*fitters, labels=[str(g) for g in order], ax=ax, rows_to_show=["At risk"], fontsize=8)
    tab = km_table(d, group, time, event, order)
    return ax, tab


def cox_fit(df, covariates, time="time", event="event", entry=None):
    """Cox PH with automatic dummy coding of categorical covariates. entry = left-truncation column.
    Returns (fitter, summary DataFrame with HR / CI / p). attrs carry n and events."""
    cols = [time, event] + list(covariates) + ([entry] if entry else [])
    d = df[cols].dropna().copy()
    if entry:
        d = d[d[entry] < d[time]]
    X = pd.get_dummies(d[list(covariates)], drop_first=True, dtype=float)
    data = pd.concat([d[[time, event]].astype(float), X], axis=1)
    if entry:
        data[entry] = d[entry].astype(float)
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data, duration_col=time, event_col=event, entry_col=entry)
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["HR", "HR_lo", "HR_hi", "p"]
    s.attrs.update(n=len(data), events=int(data[event].sum()))
    return cph, s


def hr(df, feature, time="time", event="event", entry=None, adjust=()):
    """Hazard ratio of a boolean/0-1 feature, optionally adjusted and left-truncated. NaNs on failure."""
    d = df.copy()
    d[feature] = d[feature].astype(float)
    try:
        _, s = cox_fit(d, [feature] + list(adjust), time, event, entry)
        r = s.loc[feature]
        return dict(HR=r.HR, HR_lo=r.HR_lo, HR_hi=r.HR_hi, p_cox=r.p, n_cox=s.attrs["n"])
    except Exception as e:  # convergence / separation
        return dict(HR=np.nan, HR_lo=np.nan, HR_hi=np.nan, p_cox=np.nan, n_cox=np.nan)


def compare(surv, status: pd.Series, labels=("WT", "altered"), time="time", event="event", **hr_kw):
    """Two-group summary for a boolean status Series: KM table + log-rank + HR. Returns (df, table)."""
    d = surv.join(status.rename("feat"), how="inner").dropna(subset=["feat", time, event])
    d["group"] = np.where(d.feat.astype(bool), labels[1], labels[0])
    tab = km_table(d, "group", time, event, order=list(labels))
    h = hr(d, "feat", time=time, event=event, **hr_kw)
    tab.attrs.update(h)
    for k in ["HR", "HR_lo", "HR_hi", "p_cox"]:
        tab[k] = np.nan
        tab.loc[tab.group == labels[1], k] = h[k]
    return d, tab


def gene_screen(surv, alts: Alterations, kinds=("mut",), genes=None, min_n=10, time="time", event="event",
                entry=None, adjust=(), patients=None) -> pd.DataFrame:
    """Test every gene feature (altered vs not, among patients tested for the gene) against OS.
    Groups smaller than min_n are skipped. BH-FDR over the log-rank p-values."""
    kinds = alts._kinds(kinds)
    base = surv if patients is None else surv.loc[surv.index.intersection(patients)]
    base = base.dropna(subset=[time, event])
    rows = []
    for k in kinds:
        M, T = alts.matrix(k), alts.tested_for(k)
        for g in (genes if genes is not None else M.columns):
            if g not in M.columns:
                continue
            idx = base.index.intersection(T.index[T[g]])
            st = M.loc[idx, g]
            n_alt, n_wt = int(st.sum()), int((~st).sum())
            if n_alt < min_n or n_wt < min_n:
                continue
            d = base.loc[idx].copy()
            d["feat"] = st.astype(int).values
            p = logrank_p(d, "feat", time, event)
            km_a = KaplanMeierFitter().fit(d.loc[d.feat == 1, time], d.loc[d.feat == 1, event])
            km_w = KaplanMeierFitter().fit(d.loc[d.feat == 0, time], d.loc[d.feat == 0, event])
            row = dict(feature=f"{g} {k}", gene=g, kind=k, n_alt=n_alt, n_wt=n_wt,
                       events_alt=int(d.loc[d.feat == 1, event].sum()), events_wt=int(d.loc[d.feat == 0, event].sum()),
                       median_alt=round(km_a.median_survival_time_, 1), median_wt=round(km_w.median_survival_time_, 1),
                       p_logrank=p)
            row.update(hr(d, "feat", time, event, entry, adjust))
            rows.append(row)
    res = pd.DataFrame(rows)
    if len(res):
        res["q_bh"] = multipletests(res.p_logrank, method="fdr_bh")[1]
        res = res.sort_values("p_logrank").reset_index(drop=True)
    return res


def forest_plot(res: pd.DataFrame, top=15, title="", ax=None, label_col="feature"):
    """Hazard ratios with 95% CI for the top features of a screen (log scale)."""
    r = res.dropna(subset=["HR"]).head(top).iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(r) + 1.2))
    y = np.arange(len(r))
    ax.hlines(y, r.HR_lo, r.HR_hi, color=PALETTE[0], linewidth=2)
    ax.plot(r.HR, y, "o", color=PALETTE[0], ms=7, markeredgecolor="white")
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_xscale("log")
    lo, hi = np.nanmin(r.HR_lo.clip(lower=0.05)), np.nanmax(r.HR_hi.clip(upper=50))
    ticks = [t for t in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32] if lo / 1.3 <= t <= hi * 1.3] or [0.5, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}  (n={n})" for f, n in zip(r[label_col], r.n_alt)])
    for yi, (p, q) in enumerate(zip(r.p_logrank, r.q_bh)):
        ax.text(1.01, yi, f"p={fmt_p(p)}  q={fmt_p(q)}", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2, va="center")
    ax.set_xlabel("Hazard ratio (altered vs not), 95% CI")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    return ax


def bar_freq(freq: pd.DataFrame, top=20, title="", ax=None, label_col="gene", value_col="pct"):
    """Horizontal bar chart of alteration frequencies."""
    f = freq.head(top).iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(f) + 1))
    ax.barh(f[label_col].astype(str) + " " + f["kind"] if "kind" in f else f[label_col], f[value_col], color=PALETTE[0], height=0.65)
    for i, (v, n) in enumerate(zip(f[value_col], f.get("n_altered", f[value_col]))):
        ax.text(v + 0.3, i, f"{v:.1f}% (n={n})" if "n_altered" in f else f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("% of tested patients")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    return ax


def save(dataset: str, qid: str, table: pd.DataFrame | None = None, fig=None, slug: str = ""):
    """Write results/<dataset>/<qid>[_<slug>].csv / .png."""
    d = RESULTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    name = f"{qid}_{slug}" if slug else qid
    if table is not None:
        table.to_csv(d / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
    if fig is not None:
        fig.savefig(d / f"{name}.png", bbox_inches="tight")
    return d / name


def finding(qid: str, verdict: str, note: str, store: list):
    """Collect one-line verdicts per question; printed as a table at the end of each notebook."""
    store.append(dict(question=qid, verdict=verdict, note=note))
    print(f"[{qid}] {verdict}: {note}")


# --------------------------------------------------------------------------------------
# GENIE main release loader (v20.0-public): no treatment data, OS derived from sequencing
# --------------------------------------------------------------------------------------

def _censored_number(s: pd.Series) -> pd.Series:
    """GENIE encodes censored values as '>32485', '<6570', '>89', '<18'; everything else non-numeric -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"^[<>]", "", regex=True), errors="coerce")


class GenieMain:
    """GENIE main release. Genomic files are large, so mutations / CNA / SV are loaded for a sample subset on demand."""

    def __init__(self, data_dir: Path = DATA / "genie" / "GENIE_main"):
        d = self.dir = Path(data_dir)
        pt = pd.read_csv(d / "clinical/data_clinical_patient.txt", sep="\t", comment="#", low_memory=False)
        sm = pd.read_csv(d / "clinical/data_clinical_sample.txt", sep="\t", comment="#", low_memory=False)
        self.patients = pt.rename(columns={"PATIENT_ID": "patient_id"}).set_index("patient_id", drop=False)
        sm = sm.rename(columns={"SAMPLE_ID": "sample_id", "PATIENT_ID": "patient_id"})
        age_col = "AGE_AT_SEQ_REPORT_DAYS" if "AGE_AT_SEQ_REPORT_DAYS" in sm.columns else "AGE_AT_SEQ_REPORT"
        age = _censored_number(sm[age_col])
        sm["age_seq_days"] = age if age.max() > 200 else age * 365.25     # years -> days when needed
        sm["age_seq_censored"] = sm[age_col].astype(str).str.contains(r"^[<>]", regex=True)
        self.samples_all = sm
        self.panel_of = sm.set_index("sample_id").SEQ_ASSAY_ID
        self.panel_genes = load_panel_genes()
        self._maf = d / "genomic/data_mutations_extended.txt"
        self._cna = d / "genomic/data_CNA.txt"
        self._sv = d / "genomic/data_sv.txt"

    def gene_rows(self, gene: str) -> dict:
        """Every mutation / SV row for one gene across the whole release (one pass over each file)."""
        out = {}
        chunks = []
        for ch in pd.read_csv(self._maf, sep="\t", usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short"],
                              chunksize=500_000, low_memory=False):
            chunks.append(ch[ch.Hugo_Symbol == gene])
        out["mut"] = pd.concat(chunks).rename(columns={"Hugo_Symbol": "gene", "Tumor_Sample_Barcode": "sample_id"})
        if self._sv.exists():
            sv = pd.read_csv(self._sv, sep="\t", low_memory=False)
            m = (sv.Site1_Hugo_Symbol == gene) | (sv.Site2_Hugo_Symbol == gene)
            out["sv"] = sv[m]
        else:
            out["sv"] = pd.DataFrame()
        return out

    def cna_long(self, samples) -> pd.DataFrame:
        cols = pd.read_csv(self._cna, sep="\t", nrows=0).columns
        keep = ["Hugo_Symbol"] + [c for c in cols if c in set(samples)]
        cna = pd.read_csv(self._cna, sep="\t", usecols=keep, index_col=0)
        cl = cna.stack(future_stack=True).dropna().rename("value").reset_index()
        cl.columns = ["gene", "sample_id", "value"]
        return cl

    def genomics_for(self, samples) -> tuple:
        """(mut_long, cna_long, sv_long) restricted to the given sample ids."""
        samples = set(samples)
        chunks = []
        for ch in pd.read_csv(self._maf, sep="\t", usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short"],
                              chunksize=500_000, low_memory=False):
            chunks.append(ch[ch.Tumor_Sample_Barcode.isin(samples)])
        maf = pd.concat(chunks).rename(columns={"Hugo_Symbol": "gene", "Tumor_Sample_Barcode": "sample_id"})
        mut_long = maf[maf.Variant_Classification.isin(NONSYN)].dropna(subset=["gene"])
        cna_long = self.cna_long(samples)
        if self._sv.exists():
            sv = pd.read_csv(self._sv, sep="\t", low_memory=False)
            sv = sv[sv.Sample_Id.isin(samples)]
            long = pd.concat([sv[["Sample_Id", "Site1_Hugo_Symbol", "Event_Info"]].rename(columns={"Site1_Hugo_Symbol": "gene"}),
                              sv[["Sample_Id", "Site2_Hugo_Symbol", "Event_Info"]].rename(columns={"Site2_Hugo_Symbol": "gene"})])
            sv_long = long.rename(columns={"Sample_Id": "sample_id", "Event_Info": "detail"}).dropna(subset=["gene"]).drop_duplicates()
        else:
            sv_long = pd.DataFrame(columns=["sample_id", "gene", "detail"])
        return mut_long, cna_long, sv_long

    def alterations(self, samples: pd.DataFrame, genes=None) -> Alterations:
        """samples: DataFrame[sample_id, patient_id] (one row per sample to use)."""
        mut_long, cna_long, sv_long = self.genomics_for(samples.sample_id)
        self.last_genomics = (mut_long, cna_long, sv_long)
        return build_alterations(samples[["sample_id", "patient_id"]], mut_long, cna_long, sv_long, self.panel_of, self.panel_genes, genes)

    def survival_from_sequencing(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Derived OS: months from the sequencing report (earliest sample given per patient) to death or last contact.
        INT_DOD / INT_CONTACT are days from birth; AGE_AT_SEQ_REPORT is the age at the report. Censored intervals (>89 y, <18 y) are
        kept at the censoring boundary and flagged."""
        s = samples.sort_values(["patient_id", "age_seq_days"]).drop_duplicates("patient_id").set_index("patient_id")
        p = self.patients.reindex(s.index)
        dead = p.DEAD.astype(str).str.lower().map({"true": 1, "false": 0})
        int_dod, int_contact = _censored_number(p.INT_DOD), _censored_number(p.INT_CONTACT)
        end = int_dod.where(dead == 1, int_contact)
        out = pd.DataFrame({
            "time": (end - s.age_seq_days) / 30.4375, "event": dead,
            "age_seq_years": s.age_seq_days / 365.25,
            "censored_interval": p.INT_CONTACT.astype(str).str.contains(r"^[<>]", regex=True) | p.INT_DOD.astype(str).str.contains(r"^[<>]", regex=True) | s.age_seq_censored,
            "sex": p.SEX, "center": p.CENTER, "sample_id": s.sample_id,
            "cancer_type": s.get("CANCER_TYPE"), "oncotree": s.get("ONCOTREE_CODE"), "sample_type": s.get("SAMPLE_TYPE"), "seq_year": s.get("SEQ_YEAR"),
        })
        out.loc[out.time < 0, "time"] = np.nan          # last contact before the sequencing report: not usable
        out.index.name = "patient_id"
        return out
