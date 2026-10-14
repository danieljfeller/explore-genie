"""Re-fit the headline Kaplan-Meier comparisons and export their step curves as JSON for the website.

Each entry: groups (name, n, events, median, step times t / survival s, censor times c, at-risk counts at ticks),
log-rank p, xmax, ticks, axis label. Cohort definitions mirror the notebooks exactly.
"""
import ast, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cbio_lib as cb  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "site" / "curves.json"
C = {}


def curve(df, group, order=None, time="time", event="event", xmax=None, xlabel="Months", title="", note=""):
    d = df.dropna(subset=[group, time, event]).copy()
    d[event] = d[event].astype(int)
    order = [g for g in (order or sorted(d[group].unique(), key=str)) if (d[group] == g).any()]
    xmax = float(xmax or min(d[time].max(), d[time].quantile(0.99) * 1.1))
    step = 12 if xmax <= 160 else 24
    ticks = list(range(0, int(xmax) + 1, step))
    groups = []
    for g in order:
        sub = d[d[group] == g]
        kmf = KaplanMeierFitter().fit(sub[time], sub[event])
        sf = kmf.survival_function_
        t, s = sf.index.values.astype(float), sf.iloc[:, 0].values.astype(float)
        if len(t) > 300:
            idx = np.unique(np.concatenate([[0, len(t) - 1], np.linspace(0, len(t) - 1, 300).astype(int)]))
            t, s = t[idx], s[idx]
        cens = np.sort(sub.loc[sub[event] == 0, time].values.astype(float))
        if len(cens) > 200:
            cens = cens[np.linspace(0, len(cens) - 1, 200).astype(int)]
        med = kmf.median_survival_time_
        groups.append(dict(name=str(g), n=int(len(sub)), events=int(sub[event].sum()),
                           median=None if np.isinf(med) else round(float(med), 1),
                           t=[round(float(x), 2) for x in t], s=[round(float(x), 4) for x in s],
                           c=[round(float(x), 1) for x in cens], ar=[int((sub[time] >= tk).sum()) for tk in ticks]))
    p = cb.logrank_p(d, group, time, event)
    return dict(title=title, note=note, xlabel=xlabel, groups=groups, p=None if pd.isna(p) else float(p), xmax=round(xmax, 1), ticks=ticks)


def save():
    OUT.write_text(json.dumps(C))
    print(f"  -> {len(C)} curve sets written")


def section(name):
    def deco(fn):
        try:
            print("==", name)
            fn()
        except Exception as e:  # keep going; the site build tolerates missing curves
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
        save()
        return fn
    return deco


@section("NSCLC")
def _nsclc():
    b = cb.BPC("NSCLC"); A = b.alterations(); S = b.survival_dx()
    egfr = A.status("EGFR"); egfr_ids = egfr.index[egfr]
    d, _ = cb.compare(S, egfr, labels=("EGFR wild-type", "EGFR mutant"))
    C["nsclc_n1"] = curve(d, "group", ["EGFR wild-type", "EGFR mutant"], xlabel="Months from diagnosis", title="OS from diagnosis by EGFR mutation status")
    prof6 = A.combo_groups(["EGFR", "KEAP1", "STK11"], min_n=10, wt_label="none of the three")
    order6 = ["none of the three", "EGFR", "KEAP1", "STK11", "KEAP1+STK11"]
    d6 = S.join(prof6, how="inner"); d6 = d6[d6.profile.isin(order6)]
    C["nsclc_n6"] = curve(d6, "profile", order6, xlabel="Months from diagnosis", title="OS from diagnosis by mutation profile")
    R = b.survival_regimen(pattern="|".join(cb.IO)).join(prof6, how="inner"); R = R[R.profile.isin(order6)]
    C["nsclc_n6_io"] = curve(R, "profile", order6, xlabel="Months from start of immunotherapy", title="OS from the start of an immunotherapy-containing regimen, by profile")
    osi = b.survival_regimen(exact="Osimertinib"); osi = osi[osi.index.isin(egfr_ids) & osi.met_setting]
    chemo = b.survival_regimen(pattern="|".join(cb.CHEMO), exclude="|".join(cb.EGFR_TKI)); chemo = chemo[chemo.index.isin(egfr_ids) & chemo.met_setting]
    both = osi.index.intersection(chemo.index)
    d40 = pd.concat([S.loc[osi.index.difference(both)].assign(group="received osimertinib, never chemo"),
                     S.loc[both].assign(group="received both"),
                     S.loc[chemo.index.difference(both)].assign(group="received chemotherapy, never osimertinib")])
    C["nsclc_n4_dx"] = curve(d40, "group", ["received osimertinib, never chemo", "received both", "received chemotherapy, never osimertinib"],
                             xlabel="Months from diagnosis", title="OS from diagnosis, by treatment ever received (the deck's view)")
    d4 = pd.concat([osi.assign(group="Osimertinib"), chemo.assign(group="Chemotherapy, no EGFR TKI")]).reset_index()
    C["nsclc_n4_reg"] = curve(d4, "group", ["Osimertinib", "Chemotherapy, no EGFR TKI"], xlabel="Months from regimen start",
                              title="OS from the start of the regimen (metastatic setting)")


@section("AKT1")
def _akt1():
    P = cb.Portal("brca_akt1_genie_2019"); S = P.survival()
    tl = pd.read_csv(P.dir / "data_timeline.tsv", sep="\t")
    tl["attr"] = tl.attributes.apply(lambda s: {d["key"]: d["value"] for d in ast.literal_eval(s)} if isinstance(s, str) else {})
    met_day = tl[(tl.eventType == "STATUS") & (tl.attr.apply(lambda a: a.get("STATUS") == "Metastatic Diagnosis"))].groupby("patientId").startNumberOfDaysSinceDiagnosis.min()
    S["time_met"] = S.time - met_day.reindex(S.index) / 30.4375
    S.loc[S.time_met <= 0, "time_met"] = np.nan
    S["ai_grp"] = np.where(S.AROMATASE_INHIBITOR_YN.eq("Yes"), "aromatase inhibitor", "no aromatase inhibitor")
    S["er_pos"] = S.HORMONE_RECEPTOR_STATUS.astype(str).str.startswith("ER+")
    order = ["no aromatase inhibitor", "aromatase inhibitor"]
    C["akt1_a2_init"] = curve(S, "ai_grp", order, xlabel="Months from initial diagnosis", title="OS from initial diagnosis by aromatase-inhibitor exposure (the deck's view)")
    C["akt1_a2_met"] = curve(S[S.er_pos], "ai_grp", order, time="time_met", xlabel="Months from metastatic diagnosis", title="ER+ patients: OS from metastatic diagnosis by aromatase-inhibitor exposure")
    ai = S[S.AROMATASE_INHIBITOR_YN.eq("Yes")]
    A = P.alterations()
    d, _ = cb.compare(ai, A.status("TP53"), labels=("TP53 wild-type", "TP53 mutant"), time="time_met")
    C["akt1_a1_tp53"] = curve(d, "group", ["TP53 wild-type", "TP53 mutant"], time="time_met", xlabel="Months from metastatic diagnosis", title="AI-treated: OS from metastatic diagnosis by TP53 mutation")


@section("glioma")
def _glioma():
    P = cb.Portal("glioma_dfci_2020"); sm = P.samples_all
    gbm = sm[sm.CANCER_TYPE_DETAILED == "Glioblastoma"].copy()
    gbm["tmz"] = gbm.PRIOR_TREATMENT_TEMOZOLOMIDE.eq("Yes")
    gbm = gbm.sort_values(["patient_id", "tmz"], ascending=[True, False]).drop_duplicates("patient_id").set_index("patient_id", drop=False)
    S = P.survival().loc[P.survival().index.intersection(gbm.index)].copy()
    S["tmz"] = np.where(gbm.tmz.reindex(S.index), "temozolomide before resection", "no temozolomide before resection")
    S["smp_time"] = pd.to_numeric(gbm.SMP_OS_MOS.reindex(S.index), errors="coerce")
    order = ["no temozolomide before resection", "temozolomide before resection"]
    C["glioma_g1_dx"] = curve(S, "tmz", order, xmax=100, xlabel="Months from diagnosis", title="Glioblastoma: OS from diagnosis (the deck's view)")
    C["glioma_g1_sample"] = curve(S, "tmz", order, time="smp_time", xmax=100, xlabel="Months from sample collection", title="Glioblastoma: OS from the date the sample was collected")
    A = P.alterations(samples=gbm[["sample_id", "patient_id"]])
    d, _ = cb.compare(S, A.status("CDKN2A", "homdel"), labels=("CDKN2A intact", "CDKN2A deep deletion"))
    C["glioma_g2_cdkn2a"] = curve(d, "group", ["CDKN2A intact", "CDKN2A deep deletion"], xmax=100, xlabel="Months from diagnosis", title="Glioblastoma: OS from diagnosis by CDKN2A deep deletion")


@section("bladder")
def _bladder():
    b = cb.BPC("BLADDER"); S = b.survival_dx(); A = b.alterations()
    onc = S.index.to_series().map(b.samples.cpt_oncotree_code)
    hist_code = pd.to_numeric(S.get("naaccr_histology_cd", pd.Series(index=S.index, dtype=float)), errors="coerce")
    hist_text = S.get("ca_histology", pd.Series(index=S.index, dtype=object)).astype(str)
    uro = onc.isin({"BLCA", "UTUC", "UCU", "BLSC", "URCC", "UCA", "UPA", "BLAD"}) | hist_code.between(8120, 8131) | hist_text.str.contains("rothelial|ransitional", na=False)
    U = S[uro]
    cdk = (A.status("CDKN2A", "homdel") | A.status("CDKN2A", "mut") | A.status("CDKN2B", "homdel") | A.status("CDKN2B", "mut"))
    d, _ = cb.compare(U, cdk, labels=("CDKN2A/B intact", "CDKN2A/B altered"))
    C["bladder_b1"] = curve(d, "group", ["CDKN2A/B intact", "CDKN2A/B altered"], xlabel="Months from diagnosis", title="Urothelial carcinoma: OS from diagnosis by CDKN2A/B alteration")
    R = b.survival_regimen(pattern="Cisplatin|Carboplatin"); R = R[R.met_setting & R.index.isin(U.index)].join(cdk.rename("feat"), how="inner")
    R["group"] = np.where(R.feat, "CDKN2A/B altered", "CDKN2A/B intact")
    C["bladder_b2_platinum"] = curve(R, "group", ["CDKN2A/B intact", "CDKN2A/B altered"], xlabel="Months from start of platinum chemotherapy", title="OS from the start of platinum chemotherapy, by CDKN2A/B status")


@section("BrCa")
def _brca():
    b = cb.BPC("BrCa"); S = b.survival_dx(); A = b.alterations()
    grp = A.combo_groups(["PIK3CA", "TP53"], min_n=1, wt_label="neither").replace({"PIK3CA": "PIK3CA only", "TP53": "TP53 only", "PIK3CA+TP53": "both"})
    d1 = S.join(grp, how="inner")
    order4 = ["neither", "PIK3CA only", "TP53 only", "both"]
    C["brca_br1"] = curve(d1, "profile", order4, xlabel="Months from diagnosis", title="OS from diagnosis by PIK3CA / TP53 status")
    hr = d1[(d1.bca_subtype == "HR+, HER2-") & d1.profile.isin(["PIK3CA only", "TP53 only"])]
    C["brca_br1_hrpos"] = curve(hr, "profile", ["PIK3CA only", "TP53 only"], xlabel="Months from diagnosis", title="HR+/HER2- only: PIK3CA-only vs TP53-only")
    Rc = b.survival_regimen(exact="Fulvestrant, Palbociclib"); Rc = Rc[Rc.met_setting].join(grp, how="inner")
    C["brca_br6"] = curve(Rc[Rc.profile.isin(["PIK3CA only", "TP53 only"])], "profile", ["PIK3CA only", "TP53 only"], xlabel="Months from start of fulvestrant + palbociclib", title="OS from the start of fulvestrant + palbociclib")
    Rf = b.survival_regimen(exact="Fulvestrant"); Rf = Rf[Rf.met_setting].join(grp, how="inner")
    C["brca_br5"] = curve(Rf[Rf.profile.isin(["PIK3CA only", "TP53 only"])], "profile", ["PIK3CA only", "TP53 only"], xlabel="Months from start of fulvestrant monotherapy", title="OS from the start of fulvestrant monotherapy")


@section("CRC")
def _crc():
    b = cb.BPC("CRC"); S = b.survival_dx(); A = b.alterations()
    rg = b.regimens
    leu_start = rg[rg.regimen_drugs.fillna("").str.contains("Leucovorin", case=False)].groupby("record_id").dx_reg_start_int_mos.min()
    samp = b.samples_all.copy()
    samp["collected_mos"] = samp.dx_path_proc_cpt_mos
    samp["leu_start"] = samp.patient_id.map(leu_start)
    samp["timing_leu"] = np.select([samp.leu_start.isna(), samp.collected_mos < samp.leu_start], ["no leucovorin regimen", "biopsy before leucovorin"], default="biopsy after leucovorin")
    first = samp.sort_values(["patient_id", "collected_mos"]).drop_duplicates("patient_id").set_index("patient_id")
    S["timing_leu"] = first.timing_leu.reindex(S.index)
    S["time_from_sample"] = S.time - first.collected_mos.reindex(S.index)
    order = ["biopsy before leucovorin", "biopsy after leucovorin", "no leucovorin regimen"]
    C["crc_c1_dx"] = curve(S, "timing_leu", order, xlabel="Months from diagnosis", title="OS from diagnosis by biopsy timing (the deck's view)")
    two = S[S.timing_leu.isin(order[:2])]
    C["crc_c1_sample"] = curve(two, "timing_leu", order[:2], time="time_from_sample", xlabel="Months from biopsy", title="OS from the date of the biopsy")
    prof = A.combo_groups(["KRAS", "TP53", "APC"], min_n=10, wt_label="none of the three")
    order4 = [g for g in ["none of the three", "KRAS", "TP53", "APC", "KRAS+TP53", "KRAS+APC", "TP53+APC", "KRAS+TP53+APC"] if g in set(prof)]
    d4 = S.join(prof, how="inner")
    C["crc_c4"] = curve(d4[d4.profile.isin(order4)], "profile", order4, xlabel="Months from diagnosis", title="OS from diagnosis by KRAS / TP53 / APC profile")


@section("PANC")
def _panc():
    b = cb.BPC("PANC"); S = b.survival_dx(); A = b.alterations()
    kras = A.status("KRAS"); kras_ids = kras.index[kras]
    d, _ = cb.compare(S, kras, labels=("KRAS wild-type", "KRAS mutant"))
    C["panc_p1_kras"] = curve(d, "group", ["KRAS wild-type", "KRAS mutant"], xmax=72, xlabel="Months from diagnosis", title="OS from diagnosis by KRAS mutation")
    for g, k, lab in [("CDKN2A", "homdel", "CDKN2A deep deletion"), ("SMAD4", "homdel", "SMAD4 deep deletion")]:
        d, _ = cb.compare(S.loc[kras_ids], A.status(g, k), labels=(f"{g} intact", lab))
        C[f"panc_p1_{g.lower()}"] = curve(d, "group", [f"{g} intact", lab], xmax=72, xlabel="Months from diagnosis", title=f"KRAS-mutant: OS from diagnosis by {lab}")


@section("prostate")
def _prostate():
    b = cb.BPC("Prostate"); S = b.survival_dx(); A = b.alterations()
    erg = A.status("ERG", "alt"); erg_ids = erg.index[erg]
    ar = (A.status("AR", "amp") | A.status("AR", "mut")).rename("AR")
    pten = (A.status("PTEN", "homdel") | A.status("PTEN", "mut")).rename("PTEN")
    st = pd.concat([ar, pten], axis=1, join="inner").loc[lambda x: x.index.isin(erg_ids)]
    st["profile"] = np.select([st.AR & st.PTEN, st.AR, st.PTEN], ["AR + PTEN", "AR co-altered", "PTEN co-altered"], default="neither")
    d = S.join(st.profile, how="inner")
    C["prostate_pr2"] = curve(d, "profile", ["neither", "AR co-altered", "PTEN co-altered", "AR + PTEN"], xlabel="Months from diagnosis", title="ERG-altered: OS from diagnosis by AR / PTEN co-alteration")
    d, _ = cb.compare(S.loc[erg_ids], A.status("TP53"), labels=("TP53 wild-type", "TP53 mutant"))
    C["prostate_pr3_tp53"] = curve(d, "group", ["TP53 wild-type", "TP53 mutant"], xlabel="Months from diagnosis", title="ERG-altered: OS from diagnosis by TP53 mutation")


@section("MBC")
def _mbc():
    P = cb.Portal("mbc_genie_2020"); S = P.survival(); A = P.alterations()
    grp = A.combo_groups(["PIK3CA", "TP53"], min_n=1, wt_label="neither").replace({"PIK3CA": "PIK3CA only", "TP53": "TP53 only", "PIK3CA+TP53": "both"})
    d = S.join(grp, how="inner")
    C["mbc_m1"] = curve(d[d.profile.isin(["PIK3CA only", "TP53 only"])], "profile", ["PIK3CA only", "TP53 only"], xlabel="Months from metastatic diagnosis", title="OS from metastatic diagnosis: PIK3CA-only vs TP53-only (independent DFCI cohort)")


@section("GENIE v20")
def _v20():
    G = cb.GenieMain()
    alk = G.gene_rows("ALK")
    sv_s = set(alk["sv"].Sample_Id) if len(alk["sv"]) else set()
    alk_samples = G.samples_all[G.samples_all.sample_id.isin(sv_s)].copy()
    S = G.survival_from_sequencing(alk_samples)
    S["nsclc"] = np.where(S.cancer_type.eq("Non-Small Cell Lung Cancer"), "non-small cell lung cancer", "other cancer types")
    S = S[S.time.notna() & S.event.notna()]
    C["v20_v1"] = curve(S, "nsclc", ["non-small cell lung cancer", "other cancer types"], xlabel="Months from sequencing report", title="ALK-fusion patients: derived OS from the sequencing report")
    nsclc = alk_samples[alk_samples.CANCER_TYPE == "Non-Small Cell Lung Cancer"].sort_values(["patient_id", "age_seq_days"]).drop_duplicates("patient_id")
    A = G.alterations(nsclc[["sample_id", "patient_id"]])
    Sn = S[S.index.isin(A.patients)].copy()
    d, _ = cb.compare(Sn, A.status("TP53"), labels=("TP53 wild-type", "TP53 mutant"))
    C["v20_v2_tp53"] = curve(d, "group", ["TP53 wild-type", "TP53 mutant"], xlabel="Months from sequencing report", title="ALK-fusion NSCLC: derived OS by TP53 mutation")
    d, _ = cb.compare(Sn, A.status("CDKN2A", "homdel"), labels=("CDKN2A intact", "CDKN2A deep deletion"))
    C["v20_v2_cdkn2a"] = curve(d, "group", ["CDKN2A intact", "CDKN2A deep deletion"], xlabel="Months from sequencing report", title="ALK-fusion NSCLC: derived OS by CDKN2A deep deletion")
    res_pat = r"G1202R|L1196M|I1171[A-Z]|F1174[A-Z]|G1269A|C1156Y|L1198F|D1203N|S1206[A-Z]|E1210K|F1245[A-Z]|G1128A|T1151|L1152[A-Z]"
    alk_mut = alk["mut"][alk["mut"].sample_id.isin(nsclc.sample_id)]
    resist = alk_mut[alk_mut.HGVSp_Short.fillna("").str.contains(res_pat, regex=True)]
    Sn["resist"] = np.where(Sn.sample_id.isin(resist.sample_id), "ALK resistance mutation present", "no resistance mutation")
    C["v20_v4"] = curve(Sn, "resist", ["no resistance mutation", "ALK resistance mutation present"], xlabel="Months from sequencing report", title="ALK-fusion NSCLC: derived OS by presence of an ALK-inhibitor resistance mutation")


print("done:", sorted(C))
