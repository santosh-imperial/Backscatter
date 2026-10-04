"""Render fixed E39S evidence, without treating crop CV as independent validation."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import CLASSES

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/"output/evaluation"
LABELS={"legacy_l1":"Ungated L1 · frozen-procedure reference", "gated_l1":"Gated L1 · matched comparator",
        "gated_l2":"Gated L2 · M1 primary", "gated_lda":"Gated LDA · comparator",
        "appearance_l2":"Appearance-only L2 · comparator", "augmented_l2":"Gated + appearance L2 · M2 primary"}


def embed(path):
    mime="image/jpeg" if Path(path).suffix.lower() in {".jpg",".jpeg"} else "image/png"
    return f"data:{mime};base64,"+base64.b64encode(Path(path).read_bytes()).decode()


def table(frame):
    return frame.to_html(index=False,border=0,classes="data",float_format=lambda v:f"{v:.3f}",escape=True)


def main():
    summary=pd.read_csv(OUT/"known_summary.csv")
    feedback=pd.read_csv(OUT/"feedback_development_predictions.csv")
    sensitivity=pd.read_csv(OUT/"appearance_oof_sensitivity.csv")
    drivers=pd.read_csv(OUT/"feedback_development_drivers.csv")
    receipt=json.loads((HERE/"output/appearance_extraction_receipt.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    colors=["#8597a2","#8597a2","#2a78d6","#8597a2","#8597a2","#d27624"]
    x=np.arange(len(summary))
    short=["Ungated L1","Gated L1","Gated L2","Gated LDA","Appearance L2","Augmented L2"]
    for ax,key,title in zip(axes,["balanced_accuracy","balanced_logloss"],["Balanced accuracy (higher)","Balanced log loss (lower)"]):
        ax.bar(x,summary[key],color=colors)
        ax.set_xticks(x,short,rotation=30,ha="right",fontsize=8)
        ax.set_title(title,loc="left",fontsize=11)
        ax.set_ylabel("Crop-held-out development diagnostic")
    axes[0].axhline(1/3,color="#64717a",ls="--",lw=1)
    axes[0].set_ylim(0,1)
    fig.suptitle("31 supplied crops · shared source parents may cross folds",fontsize=12)
    fig.tight_layout(); fig.savefig(OUT/"comparison.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.6))
    for ax,name in zip(axes,["legacy_l1","gated_l2","augmented_l2"]):
        cm=pd.read_csv(OUT/f"confusion_{name}.csv",index_col=0).to_numpy(int)
        ax.imshow(cm,cmap="Blues",vmin=0,vmax=17)
        for i in range(3):
            for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>8 else "#122f42")
        ax.set_xticks(range(3),["B1","B2","B3"]); ax.set_yticks(range(3),["B1","B2","B3"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Supplied label")
        ax.set_title(LABELS[name].split(" · ")[0],fontsize=10)
    fig.suptitle("Out-of-fold confusion · counts are crops, not verified independent sources",fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"confusions.png",dpi=150); plt.close(fig)
    display=summary[["candidate","n_correct","n_sites","balanced_accuracy",*[f"recall_{c}" for c in CLASSES],"balanced_logloss","balanced_brier"]].copy()
    display.candidate=display.candidate.map(LABELS)
    display.rename(columns={f"recall_{c}":"Recall "+c.replace("_"," ") for c in CLASSES},inplace=True)
    failures=[]
    for name,part in feedback.groupby("candidate",sort=False):
        for _,row in part.iterrows():
            failures.append({"Model":LABELS[name],"Site":row.site,"Truth":row.true_batch,"Bet":row.predicted_batch,
                             "Bet score":row["score_"+row.predicted_batch],"Correct":bool(row.correct)})
    nuisance=sensitivity.groupby("variant",sort=False).agg(n_crops=("site","size"),changed_bets=("bet_changed","sum"),
        median_max_score_change=("max_absolute_score_change","median"),max_score_change=("max_absolute_score_change","max")).reset_index()
    source_link='<a href="../source_crop_audit/findings.md">Source-overlap audit (minimum-dependence diagnostic)</a>'
    response=""
    for row in receipt["response_maps"]:
        path=ROOT/row["path"]
        response+=f'<figure><img src="{embed(path)}" alt="Actual three-detector raw crops and filter response maps"><figcaption>{html.escape(path.name)} · actual sampled crop; intensity display is not a phase annotation.</figcaption></figure>'
    primary=drivers[(drivers.candidate=="augmented_l2")&(drivers["rank"]<=4)]
    attribution=table(primary[["site","predicted_batch","comparison_batch","feature","contribution","observed"]])
    redundancy=pd.read_csv(HERE/"output/appearance_features.csv")
    known=pd.read_csv(OUT/"known_measurements.csv")
    names=[c for c in redundancy if c.startswith("appearance_")]
    corr=known[names+["corr_len_px","fft_slope"]].corr(method="spearman")
    corr.loc[names,["corr_len_px","fft_slope"]].reset_index(names="feature").to_csv(OUT/"bse_anchor_redundancy.csv",index=False)
    maxcorr=corr.loc[names,["corr_len_px","fft_slope"]].abs().max()
    overview=f"Strongest absolute rank correlation across the 24 new descriptors: {maxcorr['corr_len_px']:.3f} with BSE correlation length; {maxcorr['fft_slope']:.3f} with BSE FFT slope. This is descriptive redundancy, not proof of novelty or material specificity."
    css="""*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:#193448;background:#f3f6f8;margin:0}main{max-width:1160px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{margin-top:30px;font-size:23px}.notice{background:#fff2d8;border-left:5px solid #d27624;padding:18px}section{background:white;padding:22px;margin:20px 0;border:1px solid #d9e4eb;border-radius:8px}img{width:100%;height:auto}figure{margin:20px 0}figcaption,.small{font-size:13px;color:#536a79}.scroll{overflow:auto}.data{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:left}th{background:#edf3f6}code{background:#edf3f6;padding:2px 4px}a{color:#285bb2}"""
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E39S classifier and appearance audit</title><style>{css}</style><main>
<h1>Classification without phase annotations</h1><p>E39S / D59S · fixed six candidates · 31 known crops and three revealed failure diagnostics.</p>
<div class="notice"><strong>Interpretation changed by the organiser's provenance.</strong> These are real electrode-image crops arranged into artificial visual batches, from around 15 source images. Parent IDs are unknown. Related crops may cross training/test folds. These results do not establish source-held-out accuracy, actual supplier-lot identity, manufacturing outcomes or independent significance. Batch 3 remains the challenge reference.</div>
<section><h2>Bounded model comparison</h2><p>M1 primary: complete-gated L2 multinomial. M2 primary incremental candidate: the same model with 24 fixed mask-independent image descriptors. Comparators were fixed before scores. Median imputation, scaling and logistic regularisation selection use training folds only. No feature or model-winner search, height, raw detector means or quality-flag predictors.</p><div class="scroll">{table(display)}</div><img src="{embed(OUT/'comparison.png')}" alt="Fixed candidate balanced accuracy and balanced log loss"><img src="{embed(OUT/'confusions.png')}" alt="Crop out-of-fold confusion matrices"><p>No new IID permutation p-values or confidence intervals. The matched L1 scores reproduce E37S; its historical evidence remains preserved.</p></section>
<section><h2>Revealed first-drop failure diagnostics</h2><p>Labels were already known when this protocol was fixed. These rows are development evidence, never a second unseen test or a tuning objective. Every crop receives a bet. Scores are uncalibrated balanced-target model outputs; they are not probabilities of correctness.</p><div class="scroll">{table(pd.DataFrame(failures))}</div><h2>What drives the M2 candidate's bets</h2><p>Exact linear decision-function contributions to the chosen class versus Batch 3 (or runner-up for a baseline bet). Positive contributions favour the chosen class. Intercepts and all terms are saved in CSV. Imputed terms are unobserved model inputs, not evidence measured in the image; contributions are noncausal.</p><div class="scroll">{attribution}</div></section>
<section><h2>Acquisition challenges</h2><p>Original outer fits score the withheld crop's transformed appearance descriptors. All 31 crops and three channels use the same original trim and tile positions. The appearance-only model tests all its inputs; this is not an end-to-end perturbation test of the old mask-dependent pipeline. Gain/offset is nonclipping floating-point; gamma and 32-bin quantisation may still change signature values.</p><div class="scroll">{table(nuisance)}</div><p>{overview}</p></section>
<section><h2>Actual-image visual references</h2><p>Three detectors × eight descriptors: Gaussian scale-energy fractions, spatial IQR and image-gradient axial moments/coherence. Gaussian bands overlap; they are not orthogonal spectral bins. Scales are 2/8/32 px with fixed filter context, 3×3 pooled 512 px windows. Measured union coverage is {100*receipt['measurement_coverage_range'][0]:.2f}–{100*receipt['measurement_coverage_range'][1]:.2f}%. Windows increase coverage, not sample size. These measures describe acquisition-sensitive image appearance; no chemical/phase, particle-axis or battery-performance claim is validated.</p>{response}</section>
<section><h2>Provenance and next decision</h2><p>Frozen v2 submissions and QC are preserved; all new fits use only the original 31 known labels. No automatic promotion or 34-site submission fit. Source-image mapping is the next validation requirement; {source_link}. Non-overlap cannot establish independence.</p><p><a href="protocol.md">Fixed protocol</a> · <a href="output/appearance_definition.json">Exact feature definitions</a> · <a href="output/evaluation/receipt.json">Model receipt</a> · <a href="output/evaluation/known_oof_predictions.csv">All crop scores</a> · <a href="output/evaluation/feedback_development_drivers.csv">All failure drivers</a></p><p class="small">Generated scientific charts and image references checked separately. Native browser layout is unverified: prior browser policy blocks local-file navigation.</p></section></main></html>'''
    (HERE/"report.html").write_text(page)
    print("Wrote report.html, two scientific charts and the BSE redundancy table")


if __name__=="__main__": main()
