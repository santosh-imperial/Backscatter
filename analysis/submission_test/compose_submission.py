"""Compose the organiser-facing assignment table for a test drop from the categoriser hand-off (E31 model) and the QC
summary. Run after:
  python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2 --score <drop> --out analysis/submission_test/categoriser
  python3 -m polaron_qc.report Dataset/Batch_3 <drop> reports/qc_<drop>.html --summary analysis/submission_test/qc_<drop>_summary.json
Writes predictions.csv (one row per sample) and submission.md. Predictions are frozen by committing both before the truth arrives.
Confidence = top model probability of the pre-registered primary (combined family), labelled uncalibrated (E31: top bin 0.83 vs
observed 0.67); the qualitative band uses fixed cut-offs declared here, not tuned on the drop.
"""
import os, sys, json, hashlib, re
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from polaron_qc import categorise as _C
PRIMARY = _C.PRIMARY_FAMILY   # "combined" for the frozen E33 drop (v1), "material" from D52 on
OUT = os.path.join(ROOT, "analysis", "submission_test")
# bands = the calibration bins already used in E31 (analysis/categoriser/findings.md): [0.7, 1] observed LOO accuracy 0.67; [0.5, 0.7) 0.77; [1/3, 0.5) chance-like
BANDS = [(0.70, "moderate (E31 top bin: observed LOO accuracy 0.67 at mean model probability 0.83)"), (0.50, "low (E31 middle bin: observed 0.77 at mean 0.60)"), (0.0, "very low — a bet, as requested (E31 bottom bin)")]

def band(p):
    for cut, lab in BANDS:
        if p >= cut: return lab
    return BANDS[-1][1]

def main(drop_name="Hackathon-Polaron-test"):
    st = pd.read_csv(os.path.join(OUT, "categoriser", "site_table.csv"))
    q = st[st.role.astype(str).str.contains("score|query|new|test", case=False, regex=True)] if "role" in st else st
    if len(q) == 0:
        q = st[~st.batch.isin(["Batch_1", "Batch_2", "Batch_3"])]
    ood_pct = [c for c in st.columns if c.startswith("ood_pct") and c.endswith("__morph")][:1]
    ood_exc = [c for c in st.columns if c.startswith("ood_exceed") and c.endswith("__morph")][:1]
    rows = []
    for r in q.itertuples():
        fam = PRIMARY if f"mp_Batch_1__{PRIMARY}" in st.columns else "combined"
        probs = {b: float(getattr(r, f"mp_{b}__{fam}")) for b in ("Batch_1", "Batch_2", "Batch_3")}
        assigned = max(probs, key=probs.get); p = probs[assigned]
        second = sorted(probs.items(), key=lambda kv: -kv[1])[1]
        rows.append(dict(sample=r.site, assigned_batch=assigned, confidence_model_probability=round(p, 3), confidence_band=band(p),
                         runner_up=f"{second[0]} ({second[1]:.2f})",
                         p_Batch_1=round(probs["Batch_1"], 3), p_Batch_2=round(probs["Batch_2"], 3), p_Batch_3=round(probs["Batch_3"], 3),
                         morphology_only_argmax=getattr(r, "argmax__morph", ""), acquisition_only_argmax=getattr(r, "argmax__acq", ""),
                         drivers=getattr(r, f"cat_top_features__{fam}", ""),
                         baseline_ood_percentile_morphology=(round(float(getattr(r, ood_pct[0])), 1) if ood_pct else np.nan),
                         beyond_baseline_max=(bool(getattr(r, ood_exc[0])) if ood_exc else None),
                         acquisition_group=getattr(r, "acquisition_group", ""), low_contrast=bool(getattr(r, "bright_low_contrast", False)),
                         grey_pore=bool(getattr(r, "grey_pore", False))))
    P = pd.DataFrame(rows).sort_values("sample"); P.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    h = hashlib.sha256(open(os.path.join(OUT, "predictions.csv"), "rb").read()).hexdigest()[:16]
    qc = None
    qp = os.path.join(OUT, f"qc_{drop_name}_summary.json")
    for cand in (qp, os.path.join(OUT, "qc_test_drop_summary.json")):
        if os.path.exists(cand): qc = json.load(open(cand)); break
    lines = [f"# Test drop `{drop_name}` — batch assignment per sample (frozen before the truth)", "",
             f"Predictions file: `analysis/submission_test/predictions.csv` (sha256 prefix `{h}`). Model: `polaron_qc.categorise`, pre-registered primary (morphology + ETD/Inlens texture + acquisition statistics), fitted on the 31 known sites; nothing was tuned on the drop. "
             "Confidence is the model probability of the assigned batch and is **not calibrated** (leave-one-site-out on the known sites: top-bin mean 0.83 vs observed accuracy 0.67; balanced accuracy 0.66, p 0.005). The organisers asked for a bet with a confidence and an explanation; this is it, with the uncertainty stated.", "",
             "| sample | assigned batch | confidence (model prob.) | band | runner-up | drivers (sign vs the Batch 3 baseline) | baseline OOD percentile (morphology) | flags |", "|---|---|---|---|---|---|---|---|"]
    for r in P.itertuples():
        fl = ", ".join(x for x, on in (("low contrast", r.low_contrast), ("grey pore", r.grey_pore)) if on) or r.acquisition_group
        lines.append(f"| {r.sample} | **{r.assigned_batch}** | {r.confidence_model_probability:.2f} | {r.confidence_band} | {r.runner_up} | {r.drivers} | {r.baseline_ood_percentile_morphology} {'(beyond baseline max)' if r.beyond_baseline_max else ''} | {fl} |")
    lines += ["", "## How to read it", "",
              "- **Assigned batch** = argmax of the model probabilities of the primary categoriser; the morphology-only and acquisition-only argmax are in the CSV so the reader can see whether the bet rests on microstructure, on imaging statistics, or on both (on the known sites, morphology alone is at chance — the model is a batch fingerprint).",
              "- **Drivers** are the top contributing features for the assigned class, with sign relative to the Batch 3 baseline (median ± MAD): this is the 'in what way different from the baseline' answer per sample. Texture and acquisition drivers are labelled acquisition-sensitive in `analysis/categoriser/findings.md`; see `docs/batch_signatures.md` for what each batch's signature looks like on the known sites.",
              "- **Baseline OOD percentile** is the sample's k-NN robust distance to the 17 Batch 3 sites, as a percentile of the baseline's own leave-one-site-out distances (rank floor 1/18); it is a separate question from the assignment and never a probability of defect.",
              "- The batch-level QC path (`reports/qc_" + drop_name.replace("Hackathon-Polaron-", "") + "_drop.html`) was also run on the three samples as one folder; with n < 5 it abstains by design and still shows flags, local-anomaly evidence and images." ]
    if qc:  # flat summary written by `python -m polaron_qc.report --summary`
        notes = qc.get("notes") or (qc.get("meta") or {}).get("notes") or []
        lines += ["", f"QC summary of the drop as one batch (n = 3, abstains by design): verdict `{qc.get('verdict', '')}`; outcome columns {qc.get('outcome_columns', '')}; reason: {qc.get('reason', '')}; "
                      f"flags: { {k: v for k, v in (qc.get('flags') or {}).items()} if isinstance(qc.get('flags'), dict) else qc.get('flags', '')}; integration notes: {notes}"]
    open(os.path.join(OUT, "submission.md"), "w").write("\n".join(lines) + "\n")
    print(P.to_string(index=False)); print("\nsha256 prefix of predictions.csv:", h)

if __name__ == "__main__":
    main(*sys.argv[1:])
