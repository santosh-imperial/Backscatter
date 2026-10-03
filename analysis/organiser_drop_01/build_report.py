"""Render the immutable first-drop predictions; no refitting or relabelling."""
from __future__ import annotations

import base64
import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .score_folder import HERE, ROOT, FREEZE, sha, write_json
from polaron_qc import categorise as CAT, features as FE


def table(frame, fmt="{:.3f}"):
    return frame.to_html(index=False, escape=True, border=0, float_format=lambda x: fmt.format(x), na_rep="unavailable")


def figure_uri(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image = Image.open(buf).convert("RGB")
    if image.width > 1500:
        image = image.resize((1500, round(image.height*1500/image.width)), Image.Resampling.LANCZOS)
    result = BytesIO(); image.save(result, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(result.getvalue()).decode()


def crop(folder, site):
    raw = {d: FE.load_image(str(folder), site, d) for d in FE.DETECTORS}
    top, bottom = FE.bright_bands(raw["BSE"])
    trimmed = {d: a[top:len(a)-bottom if bottom else None] for d, a in raw.items()}
    height, width = trimmed["BSE"].shape
    side = min(640, height, width)
    y, x = (height-side)//2, (width-side)//2
    return raw, trimmed, (top, bottom, y, x, side)


def quality_note(feature, low, grey):
    bright = feature.startswith("bright_") or feature.startswith("inlens_particle_") or feature == "inlens_speckled_particle_frac" or feature == "etd_crack_density_particles"
    pore = feature.startswith(("pore_", "crack_")) and feature not in CAT.TEXTURE_FEATURES
    if low and bright:
        return "unreliable bright mask / interior selection"
    if grey and pore:
        return "fallback pore threshold"
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'first_run')
    args = parser.parse_args(argv)
    out = args.out
    receipt = json.loads((out / "prediction_receipt.json").read_text())
    for name, h in receipt["output_sha256"].items():
        if sha(out/name) != h:
            raise ValueError(f"First-prediction artifact changed: {name}")
    if sha(FREEZE/'models.joblib') != receipt['model_sha256']:
        raise ValueError('Frozen model changed since prediction')
    saved = joblib.load(FREEZE / "models.joblib")
    model, known = saved["models"]["combined"], saved["known"]
    sites = pd.read_csv(out / "features.csv").set_index("site", drop=False)
    scored = pd.read_csv(out / "site_table.csv").set_index("site", drop=False)
    submitted = pd.read_csv(out / "submission_sites.csv")
    registry = json.loads((ROOT / "analysis/morphology/metric_register.json").read_text())
    labels = {key:m["label"] for m in registry["metrics"] for key in m["keys"]}
    units = {key:m["units"] for m in registry["metrics"] for key in m["keys"]}
    labels.update({'ridge_p97':'ETD ridge response, 97th percentile', 'bse_p50':'BSE median intensity',
                   'inlens_p50':'Inlens median intensity', 'etd_p50':'ETD median intensity',
                   'bse_std':'BSE intensity standard deviation', 'bse_empty_bin_frac':'BSE empty histogram-bin fraction'})
    units.update({'ridge_p97':'normalised response', 'bse_p50':'grey level', 'inlens_p50':'grey level',
                  'etd_p50':'grey level', 'bse_std':'grey level', 'bse_empty_bin_frac':'fraction', 'H':'px'})
    source_folder = Path(receipt["input_directory"])
    scaler = model.pipeline["scale"]; imputer = model.pipeline["impute"]
    Zknown = scaler.transform(imputer.transform(known[model.features].to_numpy(float)))
    sections, contrast_rows, evidence = [], [], []
    colours = {"Batch_1":"#2a78d6", "Batch_2":"#eb6834", "Batch_3":"#1baf7a"}
    for _, result in submitted.iterrows():
        sid, assignment, runner = result.sample_id, result.predicted_batch, result.runner_up
        s, r = sites.loc[sid], scored.loc[sid]
        low, grey = bool(r.bright_low_contrast), bool(r.grey_pore)
        x = s[model.features].to_numpy(float)
        z = scaler.transform(imputer.transform(x[None,:]))[0]
        win, second = model.classes.index(assignment), model.classes.index(runner)
        lr = model.pipeline["lr"]
        contrast = (lr.coef_[win] - lr.coef_[second])*z
        intercept = float(lr.intercept_[win]-lr.intercept_[second])
        logits = model.pipeline.decision_function(x[None,:])[0]
        assert np.isclose(intercept+contrast.sum(), logits[win]-logits[second])
        rows=[]
        for j in np.argsort(-np.abs(contrast), kind="stable"):
            if contrast[j]==0:
                continue
            f=model.features[j]
            family=("morphology" if f in CAT.MORPH_FEATURES else "acquisition-sensitive texture" if f in CAT.TEXTURE_FEATURES else "acquisition")
            row={"feature":f, "measurement":labels.get(f,f)+' ['+units.get(f,'native units')+']',
                 "family":family, "query":x[j], "Batch_1 median":known.loc[known.batch=='Batch_1',f].median(),
                 "Batch_2 median":known.loc[known.batch=='Batch_2',f].median(), "Batch_3 median":known.loc[known.batch=='Batch_3',f].median(),
                 "toward assigned vs runner":float(contrast[j]), "quality":quality_note(f,low,grey)}
            rows.append(row);contrast_rows.append({"site":sid, "assigned":assignment, "runner_up":runner, **row})
        distance=np.sqrt(((Zknown-z)**2).sum(axis=1))
        eligible=np.flatnonzero(known.batch.to_numpy()==assignment)
        nearest=known.iloc[eligible[np.argmin(distance[eligible])]]
        reference_site=str(r.ood_nearest_ref__morph).split('|')[0].split('(')[0]
        raw, trimmed, (top,bottom,y,x0,side)=crop(source_folder,sid)
        fig,axs=plt.subplots(1,1,figsize=(12,3.5))
        axs.imshow(raw['BSE'],cmap='gray',vmin=0,vmax=255,aspect='auto')
        axs.add_patch(Rectangle((x0,top+y),side,side,fill=False,edgecolor='#d07bf9',linewidth=2))
        if top:axs.axhspan(0,top,color='#ed9b35',alpha=.3)
        if bottom:axs.axhspan(raw['BSE'].shape[0]-bottom,raw['BSE'].shape[0],color='#ed9b35',alpha=.3)
        axs.set(xlabel='image x (px)',ylabel='image y (px)',title=f'{sid}: full BSE strip; purple = fixed central crop, orange = trimmed edge bands')
        strip=figure_uri(fig)
        fig,axs=plt.subplots(1,4,figsize=(12,3.4))
        for ax,det in zip(axs[:3],FE.DETECTORS):
            ax.imshow(trimmed[det][y:y+side,x0:x0+side],cmap='gray',vmin=0,vmax=255);ax.set_title(det);ax.set_axis_off()
        pore,bright,_=FE.segment(trimmed['BSE'],float(s.th_lo),float(s.th_hi))
        assert np.isclose(pore.mean(),s.pore_frac,rtol=0,atol=1e-12), 'Overlay must reproduce measured void fraction'
        assert np.isclose(bright.mean(),s.bright_frac,rtol=0,atol=1e-12), 'Overlay must reproduce measured bright fraction'
        base=trimmed['BSE'][y:y+side,x0:x0+side]
        colour=np.repeat(base[...,None]/255,3,axis=2)
        for mask,c in ((pore,(.1,.8,1.)),(bright,(1.,.75,.1))):
            m=mask[y:y+side,x0:x0+side];colour[m]=.5*colour[m]+.5*np.array(c)
        axs[3].imshow(colour);axs[3].set_title('Void cyan / bright gold');axs[3].set_axis_off()
        query=figure_uri(fig)
        fig,axs=plt.subplots(1,3,figsize=(10,3.4))
        axs[0].imshow(base,cmap='gray',vmin=0,vmax=255);axs[0].set_title(f'Query {sid}')
        for ax,nb,ns,title in ((axs[1],nearest.batch,nearest.site,'Nearest in assigned batch'),(axs[2],'Batch_3',reference_site,'Nearest baseline morphology')):
            _,nt,(_,_,ny,nx,nside)=crop(ROOT/'Dataset'/nb,str(ns))
            ax.imshow(nt['BSE'][ny:ny+nside,nx:nx+nside],cmap='gray',vmin=0,vmax=255)
            ax.set_title(f'{title}\n{nb} / {ns}',fontsize=9)
        for ax in axs:ax.set_axis_off()
        nearest_uri=figure_uri(fig)
        warning=("Bright segmentation and mask-based texture are unreliable on this low-contrast image. The contrast-separation feature is imputed. The high baseline distance is driven by fragmented bright masks, not established material change."
                 if low else "BSE contrast stretching is flagged: intensity/texture differences can reflect the exported contrast mapping."
                 if bool(r.contrast_stretched_bse) else "No listed acquisition flag fires; this does not establish that acquisition or preparation is equivalent.")
        flagtext=escape(result.acquisition_flags)
        probability=table(pd.DataFrame([{c:result['model_score_'+c] for c in model.classes}]))
        display_columns=['measurement','family','query','Batch_1 median','Batch_2 median','Batch_3 median','toward assigned vs runner','quality']
        medians=table(pd.DataFrame(rows[:6])[display_columns],'{:.4g}')
        sensitivity=table(pd.DataFrame([{"model":'combined (primary)',"bet":assignment}, {"model":'morphology only',"bet":result.morphology_only_bet},{"model":'acquisition only',"bet":result.acquisition_only_bet}]))
        sections.append(f'''<section id="{sid}"><h2>{sid} → <span style="color:{colours[assignment]}">{assignment.replace('_',' ')}</span></h2>
        <p>Primary bet; top model score <b>{result.top_model_score:.3f}</b>, runner-up {runner.replace('_',' ')} <b>{result['model_score_'+runner]:.3f}</b>, margin <b>{result.margin:.3f}</b>. These are uncalibrated scores, not probabilities of correct assignment.</p>{probability}
        <p>Known-training-site deletion sensitivity: <b>{int(result.deletion_same_assignment)}/{int(result.deletion_fit_count)}</b> fits retain this assignment; assigned-class score range <b>{result.assigned_score_deletion_min:.3f}–{result.assigned_score_deletion_max:.3f}</b>. Fits overlap; these are not independent votes or a confidence interval.</p>
        <div class="warning"><b>Quality: {flagtext}.</b> {escape(warning)}</div>
        <h3>Why this bet rather than {runner.replace('_',' ')}?</h3>{medians}
        <p class="small">Exact linear-logit contributions for the assigned class minus its runner-up: positive supports this bet, negative opposes it; intercept difference {intercept:.3f}. Ranking is explanatory, not causal. Baseline/class medians include known quality limitations. Morphology is measured 2-D geometry; texture/intensity origins remain unresolved.</p>
        <h3>Separate baseline comparison</h3><p>Morphology distance percentile <b>{result.baseline_morphology_percentile:.1f}</b> within the 17-site Batch 3 LOO reference. Exceeds observed maximum: <b>{str(result.exceeds_observed_baseline_max).lower()}</b>. This is a descriptive rank, not calibrated membership, error probability or a release verdict.</p>
        <p>Baseline distance drivers: <code>{escape(str(r.ood_top_features__morph))}</code></p>{sensitivity}
        <h3>Image evidence</h3><img alt="{sid} full BSE strip and crop coordinates" src="{strip}"><img alt="{sid} detector crops and phase masks" src="{query}">
        <p class="small">Fixed 0–255 displays; central {side} × {side} px crop at raw (x={x0}, y={top+y}); BSE trim top/bottom={top}/{bottom} rows. Masks were measured on the full trimmed ROI before cropping. This illustration is not expert phase truth; full-site scalar measurements also use other regions.</p>
        <img alt="{sid} matched-size BSE reference examples" src="{nearest_uri}"><p class="small">Assigned-batch retrieval uses the classifier's feature space; baseline retrieval uses morphology distance. Central crops are representative examples, not matched physical locations or independent confirmation.</p></section>''')
        evidence.append(dict(site=sid,trim_top=top,trim_bottom=bottom,raw_crop_x=x0,raw_crop_y=top+y,side=side,
                             assigned_batch_nearest_site=str(nearest.site),baseline_nearest_site=reference_site,
                             linear_logit_contrast_verified=True,overlay_phase_fractions_match_original_measurements=True))
    pd.DataFrame(contrast_rows).to_csv(out/'report_driver_contrasts.csv',index=False)
    write_json(out/'report_evidence.json',evidence)
    summary=submitted[['sample_id','predicted_batch','top_model_score','runner_up','margin','deletion_same_assignment','deletion_fit_count','baseline_morphology_percentile','acquisition_flags']]
    css='body{font:16px/1.5 system-ui;color:#183445;max-width:1250px;margin:25px auto;padding:0 18px}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{border:1px solid #cbd6de;padding:7px;text-align:right}th{background:#eef3f8}td:first-child{text-align:left}section{border-top:2px solid #cbd6de;margin-top:35px;padding-top:20px}img{width:100%;height:auto}.warning{background:#fff3d7;border-left:4px solid #e0a540;padding:12px;margin:18px 0}.small{font-size:13px;color:#475d69}code{overflow-wrap:anywhere}h1,h2,h3{line-height:1.25}a{color:#1761a3}'
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>First organiser drop — batch bets</title><style>{css}</style></head><body>
    <h1>Organiser drop: {len(submitted)} batch bets</h1><p>Every sample receives a bet, as requested by the organisers. Each site has three detector images; its assignment applies to all three files. The folder may mix batches. Ground truth is pending; no accuracy on this drop is claimed.</p>
    <div class="warning"><b>Confidence qualification.</b> The model was fit on 31 known sites only (7 / 7 / 17), with no incoming-sample fitting or tuning. Known-site combined balanced accuracy is 0.66; model probabilities are uncalibrated. Acquisition-sensitive texture and session statistics contribute to its batch fingerprint. Scores and training-set sensitivity do not establish material conformance or probability of correctness.</div>
    {table(summary)}<p><a href="submission_sites.csv">Submission by sample</a> · <a href="submission_images.csv">Submission for all image files</a> · <a href="report_driver_contrasts.csv">Full driver table</a> · <a href="prediction_receipt.json">First-prediction receipt</a> · <a href="../protocol.md">Protocol</a></p>
    {''.join(sections)}<footer><p class="small">Frozen source/model snapshot: {escape(receipt['source_snapshot_sha256'])}; first predictions saved {escape(receipt['completed_utc'])}. Byte-preserved prediction files verified before rendering. No production batch QC verdict is assigned to this mixed test folder.</p></footer></body></html>'''
    (out/'report.html').write_text(page)
    assert len(re.findall(r'<section id=',page))==len(submitted)
    print(f'Wrote {out / "report.html"}; {len(submitted)} sample cards, {len(submitted)*3} evidence panels; exact contrast decompositions verified.')


if __name__ == '__main__':
    main()
