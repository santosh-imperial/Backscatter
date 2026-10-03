"""E25 feasibility audit: directional image texture is not graphite plate orientation.

Run from the repository root with /opt/anaconda3/bin/python3
  -m analysis.battery.run_orientation_audit
No production extraction, cached features, raw images or verdict lists are changed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
features = None
DETECTORS = ("BSE", "ETD", "Inlens")
SIGMAS = (3, 12)
PATCH = 512
GUARD = 64
COLORS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
CASES = (("Batch_1", "f1vzngrs"), ("Batch_1", "4ih2ggld"),
         ("Batch_3", "71vgq3fw"), ("Batch_3", "hzumfsms"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_features_snapshot():
    """Private helper snapshot avoids concurrent production edits during research.

    Snapshot is kept once created for exact replay. The E25 copy is from observed
    clean commit 79bb912, before secondary extraction integration. Shared constants
    imported by that module are not used in this audit.
    """
    path=OUT/"orientation_features_snapshot.py"
    if not path.exists():path.write_bytes((ROOT/"polaron_qc/features.py").read_bytes())
    spec=importlib.util.spec_from_file_location("polaron_qc._orientation_features_snapshot",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(a, sigma):
    """Smoothed gradient tensor and axial tangent vector in image x/y coordinates.

    Angle zero denotes a horizontal *image line*, not a graphite plate. Derivative
    Gaussian sigma and tensor integration rho=4*sigma are pixels. Image-reflect
    boundaries are used. Support windows can cross BSE phase boundaries.
    """
    a = a.astype(np.float64)
    gx = ndi.gaussian_filter(a, sigma, order=(0, 1))
    gy = ndi.gaussian_filter(a, sigma, order=(1, 0))
    xx = ndi.gaussian_filter(gx*gx, 4*sigma)
    yy = ndi.gaussian_filter(gy*gy, 4*sigma)
    xy = ndi.gaussian_filter(gx*gy, 4*sigma)
    trace = xx+yy
    # The tangent is orthogonal to the dominant gradient direction.
    z = (yy-xx)-2j*xy
    return z, trace


def sums(z, trace, mask):
    mask = mask & np.isfinite(trace) & (trace > 1e-12)
    if not mask.any():
        return 0j, 0.0, 0.0, 0
    return complex(z[mask].sum()), float(trace[mask].sum()), \
        float((np.abs(z[mask])/trace[mask]).sum()), int(mask.sum())


def finish(parts):
    z = sum(x[0] for x in parts)
    tr = sum(x[1] for x in parts)
    n = sum(x[3] for x in parts)
    if tr <= 1e-12 or n == 0:
        return dict(texture_horizontal_alignment=np.nan, texture_alignment_strength=np.nan,
                    texture_axial_angle_deg=np.nan, texture_local_coherence=np.nan,
                    sampled_residual_solid_pixels=n)
    q = z/tr
    return dict(texture_horizontal_alignment=float(q.real),
                texture_alignment_strength=float(abs(q)),
                texture_axial_angle_deg=float(np.degrees(np.angle(q)/2)),
                texture_local_coherence=float(sum(x[2] for x in parts)/n),
                sampled_residual_solid_pixels=n)


def local_contrast(a, sigma=32):
    """Local standardisation, a diagnostic only, with an affine-covariant floor.

    N=(I-G32(I))/sqrt(max(G32(I²)-G32(I)²,0)+(0.05*IQR)²).
    This suppresses additive/multiplicative gain, not saturation, non-linear LUTs,
    charging relief, or spatial changes in noise. Local normalisation changes the
    measurand; it cannot establish that residual texture has a material origin.
    """
    a = a.astype(np.float64)
    mu = ndi.gaussian_filter(a, sigma)
    var = np.maximum(ndi.gaussian_filter(a*a, sigma)-mu*mu, 0)
    iqr = float(np.subtract(*np.percentile(a, [75, 25])))
    floor = max(.05*iqr, 1e-8)
    denom = np.sqrt(var+floor*floor)
    return (a-mu)/denom, var < floor*floor


def grad_sq(a, sigma=2):
    return ndi.gaussian_filter(a.astype(float), sigma, order=(0, 1))**2 + \
        ndi.gaussian_filter(a.astype(float), sigma, order=(1, 0))**2


def self_checks():
    y, x = np.mgrid[:256, :256]
    horizontal = 50+30*np.cos(y/6)
    vertical = 50+30*np.cos(x/6)
    mask = np.zeros((256, 256), bool); mask[64:-64,64:-64] = True
    h = finish([sums(*tensor(horizontal,3),mask)])
    v = finish([sums(*tensor(vertical,3),mask)])
    assert h["texture_horizontal_alignment"] > .99
    assert v["texture_horizontal_alignment"] < -.99
    assert abs(h["texture_axial_angle_deg"]) < 1e-8
    assert abs(abs(v["texture_axial_angle_deg"])-90) < 1e-8
    assert np.isnan(finish([sums(*tensor(np.ones_like(x),3),mask)])["texture_alignment_strength"])
    assert np.isnan(finish([])["texture_horizontal_alignment"])
    h2 = finish([sums(*tensor(horizontal*2+70,3),mask)])
    assert abs(h2["texture_horizontal_alignment"]-h["texture_horizontal_alignment"]) < 1e-10
    n, _ = local_contrast(horizontal)
    n2, _ = local_contrast(horizontal*1.7+23)
    assert np.allclose(n,n2,atol=1e-9)
    return dict(horizontal_and_vertical_sign=True, constant_and_empty_abstain=True,
                affine_orientation_invariance=True, affine_local_normalisation_invariance=True)


def patch_grid(H,W):
    # Deterministic, common coordinates for all detectors; patches are not n.
    return [(int(round(fy*(H-PATCH))), int(round(fx*(W-PATCH))))
            for fy in (.25,.75) for fx in (.10,.35,.65,.90)]


def raw_channels(batch,site):
    raw = {d:features.load_image(str(ROOT/"Dataset"/batch),site,d) for d in DETECTORS}
    assert len({a.shape for a in raw.values()}) == 1
    top,bot = features.bright_bands(raw["BSE"])
    sl = slice(top,raw["BSE"].shape[0]-bot if bot else None)
    return {d:a[sl] for d,a in raw.items()},top,bot


def extract():
    sites = pd.read_csv(ROOT/"analysis/morphology/output/morphology_sites.csv")
    rows, patch_rows, il_rows, mask_rows = [], [], [], []
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in
              (Path(__file__), OUT/"orientation_features_snapshot.py",
               ROOT/"analysis/morphology/output/morphology_sites.csv")}
    inputs = []
    for r in sites.itertuples():
        for d in DETECTORS:
            p = features._image_path(str(ROOT/"Dataset"/r.batch),r.site,d)
            hashes[str(Path(p).relative_to(ROOT))] = sha256(p)
            inputs.append(Path(p))
        images, top, bot = raw_channels(r.batch,r.site)
        H,W = images["BSE"].shape
        assert (H,W) == (r.H,r.W)
        parts = {(d,s,o):[] for d in DETECTORS for s in SIGMAS for o in (-5,0,5)}
        il_ss_raw=il_ss_lcn=0.; il_n=il_sat=il_floor=0
        largest_shares=[]
        for pid,(y,x) in enumerate(patch_grid(H,W)):
            patch_rows.append(dict(batch=r.batch,site=r.site,patch=pid,
                                   raw_y=top+y,raw_x=x,H=PATCH,W=PATCH,
                                   analysis_guard_px=GUARD))
            crop = {d:a[y:y+PATCH,x:x+PATCH] for d,a in images.items()}
            sm = features.smooth_bse(crop["BSE"])
            guard = np.zeros(sm.shape,bool);guard[GUARD:-GUARD,GUARD:-GUARD]=True
            masks = {}
            for o in (-5,0,5):
                pore = ndi.binary_opening(sm < r.th_lo+o)
                bright = ndi.binary_opening(sm > r.th_hi+o)
                residual = ~(pore|bright)
                masks[o] = ndi.binary_erosion(residual,iterations=8) & guard
                if o==0:
                    inner_bright = ndi.binary_erosion(bright,iterations=6)&guard
                    lab,nlab=ndi.label(residual)
                    areas=np.bincount(lab.ravel())[1:]
                    if len(areas):largest_shares.append(float(areas.max()/areas.sum()))
            for d in DETECTORS:
                for s in SIGMAS:
                    z,tr = tensor(crop[d],s)
                    for o in (-5,0,5):parts[d,s,o].append(sums(z,tr,masks[o]))
            # The same bright-mask support is used for raw/global-IQR and locally
            # standardised gradient energy; this is not the existing particle SD.
            if inner_bright.sum() >= 400:
                il=crop["Inlens"].astype(float)
                iqr=max(float(np.subtract(*np.percentile(il,[75,25]))),1e-8)
                normalised,floor=local_contrast(il)
                il_ss_raw+=float(grad_sq(il)[inner_bright].sum()/iqr**2)
                il_ss_lcn+=float(grad_sq(normalised)[inner_bright].sum())
                il_n+=int(inner_bright.sum());il_sat+=int(((il==0)|(il==255))[inner_bright].sum())
                il_floor+=int(floor[inner_bright].sum())
        for (d,s,o),a in parts.items():
            rows.append(dict(batch=r.batch,site=r.site,detector=d,sigma_px=s,rho_px=4*s,
                             threshold_offset=o,grey_pore=bool(r.grey_pore),
                             bright_low_contrast=bool(r.bright_low_contrast),
                             cracked_known=bool(r.cracked_known),**finish(a)))
        il_rows.append(dict(batch=r.batch,site=r.site,grey_pore=bool(r.grey_pore),
                            bright_low_contrast=bool(r.bright_low_contrast),
                            bright_interior_sampled_pixels=il_n,
                            inlens_sampled_raw_iqr_gradient_rms=float(np.sqrt(il_ss_raw/il_n)) if il_n else np.nan,
                            inlens_sampled_local_normalised_gradient_rms=float(np.sqrt(il_ss_lcn/il_n)) if il_n else np.nan,
                            inlens_sampled_bright_saturation_fraction=il_sat/il_n if il_n else np.nan,
                            inlens_local_normalisation_floor_fraction=il_floor/il_n if il_n else np.nan))
        mask_rows.append(dict(batch=r.batch,site=r.site,
                              residual_largest_component_share_patch_median=float(np.median(largest_shares)),
                              n_patches=len(largest_shares)))
        # Checkpoint completed site measurements before the final integrity guard.
        # A failed guard remains an unpublished/incomplete run, but is inspectable.
        for name,data in (("orientation_sites",rows),("orientation_patch_manifest",patch_rows),
                          ("orientation_inlens_sites",il_rows),("orientation_residual_mask_sites",mask_rows)):
            pd.DataFrame(data).to_csv(OUT/(name+".csv"),index=False)
        print(f"orientation audit {r.batch}/{r.site}",flush=True)
    for p,h in hashes.items():assert sha256(ROOT/p)==h, f"Input/source changed during extraction: {p}"
    pd.DataFrame(rows).to_csv(OUT/"orientation_sites.csv",index=False)
    pd.DataFrame(patch_rows).to_csv(OUT/"orientation_patch_manifest.csv",index=False)
    pd.DataFrame(il_rows).to_csv(OUT/"orientation_inlens_sites.csv",index=False)
    pd.DataFrame(mask_rows).to_csv(OUT/"orientation_residual_mask_sites.csv",index=False)
    manifest=dict(experiment="E25",source_guard_passed=True,input_hashes=hashes,
                  measurement_checks=self_checks(),extraction_complete=True)
    (OUT/"orientation_extraction_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return sites,hashes


def abs_axial_difference(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+90)%180-90)


def summarize(sites,hashes,checks):
    df=pd.read_csv(OUT/"orientation_sites.csv")
    il=pd.read_csv(OUT/"orientation_inlens_sites.csv")
    masks=pd.read_csv(OUT/"orientation_residual_mask_sites.csv")
    nominal=df[df.threshold_offset==0]
    band=[]
    for keys,g in df.groupby(["batch","site","detector","sigma_px"]):
        val=g.texture_horizontal_alignment.to_numpy()
        band.append(dict(zip(("batch","site","detector","sigma_px"),keys)),)
        band[-1].update(texture_alignment_threshold_band=float(np.nanmax(val)-np.nanmin(val)))
    pd.DataFrame(band).to_csv(OUT/"orientation_sensitivity.csv",index=False)
    agreement=[]
    for sigma,g in nominal.groupby("sigma_px"):
        wide=g.pivot(index=["batch","site"],columns="detector",values="texture_axial_angle_deg")
        for a,b in (("BSE","ETD"),("BSE","Inlens"),("ETD","Inlens")):
            for (batch,site),value in zip(wide.index,abs_axial_difference(wide[a],wide[b])):
                agreement.append(dict(batch=batch,site=site,sigma_px=int(sigma),
                                      detector_a=a,detector_b=b,axial_angle_difference_deg=float(value)))
    pd.DataFrame(agreement).to_csv(OUT/"orientation_detector_agreement.csv",index=False)
    # Confound screens use sites, never pixels or patches, and no p-value verdict.
    covariates=("H","bse_p1","bse_p50","bse_empty_bin_frac","etd_p50",
                "etd_curtain_anisotropy","etd_boundary_sharpness","etd_curtain_frac",
                "inlens_p50","inlens_empty_bin_frac","bright_sep")
    corr=[]
    def screen(g,measures,det,sigma):
        # The E18 source table already contains *_source provenance columns.
        # Select actual covariates, avoiding ambiguous/duplicate suffix columns.
        joined=g.merge(sites[["batch","site","acquisition_group",*covariates]],
                       on=["batch","site"],validate="one_to_one")
        for group in ("all","ordinary","Batch_1","Batch_2","Batch_3"):
            sub=joined if group=="all" else joined[joined.acquisition_group=="ordinary"] if group=="ordinary" else joined[joined.batch==group]
            for measure in measures:
                for cov in covariates:
                    ok=np.isfinite(sub[measure])&np.isfinite(sub[cov])
                    rho=float(spearmanr(sub.loc[ok,measure],sub.loc[ok,cov]).statistic) if ok.sum()>=4 and sub.loc[ok,cov].nunique()>1 and sub.loc[ok,measure].nunique()>1 else np.nan
                    corr.append(dict(detector=det,sigma_px=sigma,group=group,kpi=measure,
                                     covariate=cov,n_sites=int(ok.sum()),spearman_rho=rho))
    for (d,s),g in nominal.groupby(["detector","sigma_px"]):
        screen(g,["texture_horizontal_alignment","texture_alignment_strength"],d,int(s))
    screen(il,["inlens_sampled_raw_iqr_gradient_rms","inlens_sampled_local_normalised_gradient_rms"],"Inlens_local_normalisation",2)
    cdf=pd.DataFrame(corr);cdf.to_csv(OUT/"orientation_acquisition_correlations.csv",index=False)
    quality=[]
    qjoined=il.merge(sites[["batch","site","acquisition_group","inlens_p50"]],
                    on=["batch","site"],validate="one_to_one")
    for batch,g in qjoined.groupby("batch"):
        for group,v in (("bright_usable",g[~g.bright_low_contrast]),
                        ("ordinary_source_group",g[g.acquisition_group=="ordinary"])):
            for k in ("inlens_sampled_raw_iqr_gradient_rms","inlens_sampled_local_normalised_gradient_rms"):
                ok=np.isfinite(v[k])&np.isfinite(v.inlens_p50)
                rho=float(spearmanr(v.loc[ok,k],v.loc[ok,"inlens_p50"]).statistic) if ok.sum()>=4 else np.nan
                quality.append(dict(batch=batch,group=group,kpi=k,covariate="inlens_p50",
                                    n_sites=int(ok.sum()),spearman_rho=rho))
    qdf=pd.DataFrame(quality);qdf.to_csv(OUT/"orientation_inlens_quality_sensitivity.csv",index=False)
    groups=[]
    for (d,s),g in nominal.groupby(["detector","sigma_px"]):
        for group in ("Batch_1","Batch_2","Batch_3","grey_pore","ordinary"):
            sub=g[g.grey_pore] if group=="grey_pore" else g[~g.grey_pore&~g.bright_low_contrast] if group=="ordinary" else g[g.batch==group]
            groups.append(dict(detector=d,sigma_px=int(s),group=group,n_sites=len(sub),
                               horizontal_alignment_site_median=float(sub.texture_horizontal_alignment.median()),
                               alignment_strength_site_median=float(sub.texture_alignment_strength.median()),
                               local_coherence_site_median=float(sub.texture_local_coherence.median())))
    pd.DataFrame(groups).to_csv(OUT/"orientation_group_summary.csv",index=False)
    ag=pd.DataFrame(agreement)
    summary=dict(experiment="E25",n_sites=int(sites.site.nunique()),n_patches=int(len(sites)*8),
                 n_orientation_rows=len(df),n_detectors=3,derivative_sigmas_px=list(SIGMAS),
                 graphite_orientation_promoted=False,interpretation="directional image texture diagnostic only",
                 expert_plate_annotations=0,collector_direction_confirmed=False,
                 residual_mask_largest_component_share_site_median=float(masks.residual_largest_component_share_patch_median.median()),
                 detector_angle_difference_site_medians_deg={f"{a}_vs_{b}_sigma_{s}":float(g.axial_angle_difference_deg.median()) for (a,b,s),g in ag.groupby(["detector_a","detector_b","sigma_px"])},
                 local_normalisation_confound_screen=cdf[(cdf.detector=="Inlens_local_normalisation")&(cdf.covariate=="inlens_p50")][["group","kpi","n_sites","spearman_rho"]].to_dict("records"),
                 local_normalisation_quality_sensitivity=qdf[qdf.kpi=="inlens_sampled_local_normalised_gradient_rms"].to_dict("records"),
                 measurement_checks=checks,input_hashes=hashes,
                 summary_source_sha256=sha256(Path(__file__)),
                 limitations=["Eight deterministic patches per site are within-site subsamples, not new independent units.",
                              "Residual-solid centres do not delineate graphite instances; derivative and tensor windows cross phase boundaries.",
                              "Fine and coarse gradients respond to preparation striations, pores, particle edges, contrast LUTs and focus.",
                              "Phase thresholds shifted jointly by +/-5; this is a method envelope, not a confidence interval.",
                              "Image-horizontal angles are not relative to a verified collector normal.",
                              "Local contrast normalisation changes the measurand; an attenuated brightness correlation is not material validation.",
                              "Known batches were inspected before these method choices; no unseen-batch or electrochemical validation."])
    def sanitize(x):
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
        if isinstance(x,list):return [sanitize(v) for v in x]
        return x
    (OUT/"orientation_summary.json").write_text(json.dumps(sanitize(summary),indent=2,allow_nan=False)+"\n")
    return nominal,il,summary


def figures(nominal,il):
    for det in ("BSE","ETD","Inlens"):
        fig,ax=plt.subplots(figsize=(7,5))
        for batch,g in nominal[nominal.detector==det].groupby("batch"):
            wide=g.pivot(index="site",columns="sigma_px",values="texture_horizontal_alignment")
            ax.scatter(wide[3],wide[12],color=COLORS.get(batch,"#6f42c1"),label=batch)
        ax.plot([-1,1],[-1,1],"--",color="gray",linewidth=1)
        ax.set(xlim=(-1,1),ylim=(-1,1),xlabel="Fine image-texture horizontal alignment (sigma 3 px)",
               ylabel="Coarse image-texture horizontal alignment (sigma 12 px)",
               title=f"{det}: scale sensitivity; each point is one site")
        ax.legend();fig.tight_layout();fig.savefig(OUT/f"orientation_{det.lower()}_scale_sensitivity.png",dpi=140);plt.close(fig)
    # Real matched crops, with texture directions overlaid on ETD only. The grid
    # directions are not particle outlines, and annotations say so explicitly.
    fig,axes=plt.subplots(len(CASES),3,figsize=(15,18))
    for i,(batch,site) in enumerate(CASES):
        images,top,_=raw_channels(batch,site);H,W=images["BSE"].shape
        y0=(H-600)//2;x0=(W-800)//2
        for j,d in enumerate(DETECTORS):
            a=images[d][y0:y0+600,x0:x0+800]
            axes[i,j].imshow(a,cmap="gray",vmin=0,vmax=255)
            axes[i,j].set_title(f"{batch}/{site} {d}\nraw y={y0+top}, x={x0}; fixed display 0–255")
            axes[i,j].axis("off")
            if d=="ETD":
                z,tr=tensor(a,3);coh=np.abs(z)/(tr+1e-12);angle=np.angle(z)/2
                yy,xx=np.mgrid[60:a.shape[0]-60:60,60:a.shape[1]-60:60]
                q=coh[yy,xx]>.35
                dx=18*np.cos(angle[yy,xx]);dy=18*np.sin(angle[yy,xx])
                for x,y,u,v in zip(xx[q],yy[q],dx[q],dy[q]):
                    axes[i,j].plot([x-u,x+u],[y-v,y+v],color="#ffda32",lw=1.3)
    fig.suptitle("Yellow ETD marks are fine IMAGE TEXTURE directions, not graphite plate axes",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/"orientation_matched_crops.png",dpi=110);plt.close(fig)
    # A direct demonstration of normalisation on one acquisition-flagged field.
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for i,(batch,site) in enumerate((CASES[0],CASES[2])):
        ims,top,_=raw_channels(batch,site);H,W=ims["Inlens"].shape;y=(H-512)//2;x=(W-768)//2
        a=ims["Inlens"][y:y+512,x:x+768];n,floor=local_contrast(a)
        axes[i,0].imshow(a,cmap="gray",vmin=0,vmax=255);axes[i,0].set_title(f"{site}: raw Inlens (0–255)")
        axes[i,1].imshow(n,cmap="gray",vmin=-2,vmax=2);axes[i,1].set_title("Local standardisation (fixed −2…+2)")
        axes[i,2].imshow(np.sqrt(grad_sq(n)),cmap="magma",vmin=0,vmax=.4);axes[i,2].set_title("Normalised fine-gradient map (0…0.4)")
        for ax in axes[i]:ax.axis("off")
    fig.suptitle("Edges and preparation relief persist after local contrast normalisation",fontsize=13)
    fig.tight_layout();fig.savefig(OUT/"orientation_inlens_normalisation_examples.png",dpi=130);plt.close(fig)


def main():
    global features
    parser=argparse.ArgumentParser()
    parser.add_argument("--from-tables",action="store_true",help="Summarise/render completed guarded tables without re-extraction")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    features=load_features_snapshot()
    checks=self_checks()
    if args.from_tables:
        manifest=json.loads((OUT/"orientation_extraction_manifest.json").read_text())
        assert manifest["source_guard_passed"] and manifest["extraction_complete"]
        hashes=manifest["input_hashes"]
        for p,h in hashes.items():assert sha256(ROOT/p)==h,f"Manifest source/input changed: {p}"
        sites=pd.read_csv(ROOT/"analysis/morphology/output/morphology_sites.csv")
        data=pd.read_csv(OUT/"orientation_sites.csv")
        assert len(data)==31*3*2*3 and data.site.nunique()==31
        assert not data.duplicated(["batch","site","detector","sigma_px","threshold_offset"]).any()
        assert (OUT/"orientation_patch_manifest.csv").exists()
    else:
        sites,hashes=extract()
    nominal,il,summary=summarize(sites,hashes,checks)
    figures(nominal,il)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("input_hashes",)},indent=2))


if __name__=="__main__":main()
