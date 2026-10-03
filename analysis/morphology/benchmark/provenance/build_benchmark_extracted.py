"""Build the E23 ROI review pack and full-site measurement sensitivity tables.

Run: python -m analysis.morphology.build_benchmark
Manual evaluation: python -m analysis.morphology.build_benchmark --annotations PATH
Expert annotations are never fabricated or copied from the predictions.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.measure import label, regionprops_table

from polaron_qc import features
from analysis.morphology.benchmark_methods import (
    local_width, bright_hysteresis, component_summary, semantic_mask, semantic_errors,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "benchmark"
WIDTH_KEYS = ["void_local_width_d50_px", "void_local_width_d90_px", "crack_local_width_d50_px"]
CASES = {
    "f1vzngrs": ("development", "ordinary", "bright"),
    "3806gxp0": ("development", "ordinary", "bright"),
    "xgj4xftb": ("development", "ordinary / rim fragments", "bright"),
    "4ih2ggld": ("development", "bright low contrast", "bright"),
    "71vgq3fw": ("development", "grey-pore fallback", "pore"),
    "hzumfsms": ("development", "known long voids", "pore"),
    "5n1q8atc": ("held_out_site", "bright low contrast", "bright"),
    "kbdh4tri": ("held_out_site", "grey-pore fallback", "pore"),
    "ufdvpb81": ("held_out_site", "known long voids", "pore"),
    "epqdaau9": ("held_out_site", "edge band trimmed", "bright"),
    "rxax5ozo": ("held_out_site", "ordinary", "bright"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def crop_box(mask, target="bright", size=(600, 800)):
    """Purposive example around largest component, not random representative sampling."""
    props = regionprops_table(label(mask), properties=("area", "major_axis_length", "centroid"))
    if len(props["area"]):
        key = "major_axis_length" if target == "pore" else "area"
        best = int(np.argmax(props[key]))
        y, x = props["centroid-0"][best], props["centroid-1"][best]
    else:
        y, x = np.array(mask.shape) / 2
    h, w = np.minimum(size, mask.shape)
    y0 = int(np.clip(round(y-h/2), 0, mask.shape[0]-h))
    x0 = int(np.clip(round(x-w/2), 0, mask.shape[1]-w))
    return [y0, x0, y0+int(h), x0+int(w)]


def overlay(a, classes):
    rgb = np.repeat(a[:, :, None], 3, axis=2).astype(float)
    for k, color in [(0, [35, 185, 215]), (2, [245, 175, 30])]:
        use = classes == k
        rgb[use] = .55*rgb[use] + .45*np.array(color)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_roi(r, raw, bse, pore, bright, candidate, maps, band_top, band_bottom):
    split, reason, target = CASES[r.site]
    box = crop_box(pore if target == "pore" else bright, target)
    y0, x0, y1, x1 = box
    sl = np.s_[y0:y1, x0:x1]
    roi_id = f"{r.batch}_{r.site}"
    paths = {k: f"rois/{roi_id}_{k}.png" for k in ["raw", "baseline", "hysteresis", "width", "context"]}
    a = bse[sl]
    baseline, hysteresis = semantic_mask(pore[sl], bright[sl]), semantic_mask(pore[sl], candidate[sl])
    Image.fromarray(a).save(OUT / paths["raw"])
    Image.fromarray(overlay(a, baseline)).save(OUT / paths["baseline"])
    Image.fromarray(overlay(a, hysteresis)).save(OUT / paths["hysteresis"])
    width = np.where(maps["skeleton"][sl], 2*maps["distance"][sl], 0).astype(np.float32)
    # A visible colour centreline, with width values provided as a separate array.
    thick = ndi.maximum_filter(width, size=3)
    rgb = np.repeat(a[:, :, None], 3, axis=2).astype(float)
    use = thick > 0
    # Fixed 0–100 px colour range across examples, not per-ROI remapping.
    t = np.clip(thick/100, 0, 1)
    colors = np.stack([255*t, 255*(1-t), np.full(t.shape, 200)], axis=2)
    rgb[use] = colors[use]
    Image.fromarray(rgb.astype(np.uint8)).save(OUT / paths["width"])
    context = Image.fromarray(raw).convert("RGB")
    draw = ImageDraw.Draw(context)
    draw.rectangle([x0, y0+band_top, x1, y1+band_top], outline=(255,180,30), width=12)
    context.thumbnail((1000, 330))
    context.save(OUT / paths["context"])
    npz_path = f"rois/{roi_id}_predictions.npz"
    np.savez_compressed(OUT / npz_path, baseline=baseline, hysteresis=hysteresis,
                        local_width_px=width, retained_void=maps["retained"][sl])
    return dict(id=roi_id, batch=r.batch, site=r.site, split=split, reason=reason,
                selection="Purposive crop centred on the largest bright component or longest void; not a random field.",
                target=target, box_trimmed_yxyx=box,
                box_raw_yxyx=[y0+band_top, x0, y1+band_top, x1],
                H=y1-y0, W=x1-x0, band_top=int(band_top), band_bottom=int(band_bottom),
                th_lo=float(r.th_lo), th_hi=float(r.th_hi),
                bright_low_contrast=bool(r.bright_low_contrast), grey_pore=bool(r.grey_pore),
                review_status="unreviewed", measurable=None, reviewer="", notes="",
                paths=paths, predictions=npz_path)


def build():
    OUT.mkdir(exist_ok=True)
    (OUT / "rois").mkdir(exist_ok=True)
    sites = pd.read_csv(ROOT / "analysis/morphology/output/morphology_sites.csv")
    inputs = [ROOT / "analysis_cache/site_features.csv", ROOT / "analysis/morphology/output/morphology_sites.csv"]
    inputs += [ROOT / "Dataset" / r.batch / f"img_{r.site}_BSE.tif" for r in sites.itertuples()]
    sources = [Path(__file__), Path(__file__).with_name("benchmark_methods.py"),
               Path(__file__).with_name("review_template.html"), ROOT / "polaron_qc/features.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs+sources}
    manifest_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = dict(schema_version=1, experiment="E23", manifest_id=manifest_id,
                    input_hashes=hashes, threshold_offsets=[-5,0,5],
                    parameters=dict(void_min_area_px2=30, crack_major_axis_px=500,
                                    width_connectivity=8, exclude_edge_components=True,
                                    medial_axis_rng=20261003, hysteresis_half_window=5,
                                    hysteresis_propagation_connectivity=4,
                                    existing_opening_connectivity=4),
                    annotation_protocol="Independent manual polygons; unlabelled/uncertain pixels excluded. No algorithm mask is ground truth.",
                    split_protocol="Six development sites and five held-out sites. All known batches previously explored; this is a held-out annotation/method check, not unseen-batch validation. Specimen independence unknown.",
                    rois=[])
    widths, sensitivity, hysteresis_rows, connectivity = [], [], [], []
    for i,r in enumerate(sites.itertuples()):
        raw = features.load_image(str(ROOT / "Dataset" / r.batch), r.site)
        top, bottom = features.bright_bands(raw)
        bse = features.trimmed(raw)
        assert bse.shape == (r.H,r.W)
        pore,bright,sm = features.segment(bse,r.th_lo,r.th_hi)
        variants = []
        for offset in (-5,0,5):
            p = pore if offset==0 else ndi.binary_opening(sm < r.th_lo+offset, iterations=1)
            result, maps = local_width(p, return_maps=True)
            row = dict(batch=r.batch,site=r.site,threshold_offset=offset,
                       grey_pore=bool(r.grey_pore), **result)
            widths.append(row); variants.append(row)
            if offset==0:
                candidate = bright_hysteresis(sm,r.th_hi)
                base = component_summary(bright)
                cand = component_summary(candidate)
                assert np.isclose(base['frac'],r.bright_frac)
                assert np.isclose(base['d50'],r.bright_d50,equal_nan=True)
                union = (bright|candidate).sum()
                hr = dict(batch=r.batch,site=r.site,bright_low_contrast=bool(r.bright_low_contrast),
                          agreement_iou=float((bright&candidate).sum()/union) if union else np.nan,
                          changed_image_fraction=float((bright^candidate).mean()),
                          changed_bright_area_relative=float((bright^candidate).sum()/bright.sum()) if bright.any() else np.nan,
                          seeds_image_fraction=float((sm>r.th_hi+5).mean()))
                for key in base:
                    hr['baseline_'+key]=base[key];hr['hysteresis_'+key]=cand[key]
                    hr['delta_'+key]=cand[key]-base[key]
                hysteresis_rows.append(hr)
                for phase,mask,floor in [('pore',pore,30),('bright',bright,50)]:
                    c4,c8=component_summary(mask,floor,1),component_summary(mask,floor,2)
                    connectivity.append(dict(batch=r.batch,site=r.site,phase=phase,
                                             count4_per_Mpx=c4['count_per_Mpx'],count8_per_Mpx=c8['count_per_Mpx'],
                                             d50_4_px=c4['d50'],d50_8_px=c8['d50']))
                if r.site in CASES:
                    manifest['rois'].append(make_roi(r,raw,bse,pore,bright,candidate,maps,top,bottom))
            del maps
        for key in WIDTH_KEYS:
            values=np.array([x[key] for x in variants])
            sensitivity.append(dict(batch=r.batch,site=r.site,kpi=key,nominal=values[1],
                                    method_band_lo=float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
                                    method_band_hi=float(np.nanmax(values)) if np.isfinite(values).any() else np.nan))
        print(f"benchmark {i+1}/{len(sites)}: {r.batch}/{r.site}",flush=True)
    for name, rows in [('local_width_sites',widths),('local_width_sensitivity',sensitivity),
                       ('hysteresis_sites',hysteresis_rows),('connectivity_sites',connectivity)]:
        pd.DataFrame(rows).to_csv(OUT / (name+'.csv'),index=False)
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in inputs+sources}, 'Source changed during run'
    manifest['inputs_unchanged_during_run']=True
    write_json(OUT/'roi_manifest.json',manifest)
    # Never reset existing expert reviews on regeneration.
    template=OUT/'annotations_template.json'
    if not template.exists():
        write_json(template,dict(schema_version=1,manifest_id=manifest_id,
                               rois=[dict(id=r['id'],review_status='unreviewed',measurable=None,
                                          reviewer='',notes='',background='unlabelled',polygons=[]) for r in manifest['rois']]))
    render_review(manifest)


def png_data(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def render_review(manifest):
    data=json.loads(json.dumps(manifest))
    for r in data['rois']:
        r['images']={k:png_data(OUT/v) for k,v in r['paths'].items()}
    template=Path(__file__).with_name('review_template.html').read_text()
    payload=json.dumps(data,allow_nan=False).replace('</','<\\/')
    (OUT/'review.html').write_text(template.replace('__BENCHMARK_DATA__',payload))


def manual_mask(roi, annotation):
    """Rasterise independent polygons. Unlabelled is excluded, never assumed solid."""
    background=annotation.get('background','unlabelled')
    if background not in ('unlabelled','solid'):
        raise ValueError('Unknown background class')
    image=Image.new('L',(roi['W'],roi['H']),1 if background=='solid' else 255)
    draw=ImageDraw.Draw(image)
    for polygon in annotation.get('polygons',[]):
        c=polygon['class_id']; points=np.asarray(polygon['points'],float)
        if c not in (0,1,2,255) or points.ndim!=2 or points.shape[1]!=2 or len(points)<3:
            raise ValueError('Invalid annotation polygon')
        if not np.isfinite(points).all() or np.any(points<0) or np.any(points[:,0]>=roi['W']) or np.any(points[:,1]>=roi['H']):
            raise ValueError('Annotation polygon outside ROI')
        draw.polygon([tuple(p) for p in points],fill=c)
    return np.asarray(image)


def evaluate_annotations(path):
    manifest=json.loads((OUT/'roi_manifest.json').read_text())
    annotations=json.loads(Path(path).read_text())
    if annotations.get('manifest_id')!=manifest['manifest_id']:
        raise ValueError('Annotations belong to a different input/method manifest; do not silently reuse them.')
    by_id={r['id']:r for r in manifest['rois']}
    rows=[]; seen=set()
    for a in annotations['rois']:
        rid=a['id']
        if rid not in by_id or rid in seen:
            raise ValueError('Unknown or duplicate ROI ID')
        seen.add(rid);r=by_id[rid]
        if a.get('review_status')!='reviewed':
            continue
        if not a.get('reviewer','').strip() or a.get('measurable') not in (True,False):
            raise ValueError('A reviewed ROI needs reviewer name and explicit measurability.')
        if not a['measurable']:
            rows.append(dict(roi=rid,site=r['site'],split=r['split'],status='expert_unmeasurable',method='none'))
            continue
        manual=manual_mask(r,a)
        if not (manual!=255).any():
            raise ValueError('Reviewed, measurable ROI has no manually labelled pixels')
        predictions=np.load(OUT/r['predictions'])
        for method in ['baseline','hysteresis']:
            errors=semantic_errors(predictions[method],manual)
            out=dict(roi=rid,site=r['site'],split=r['split'],status='expert_scored',method=method,**errors)
            # Size/width require a fully specified mask. Partial polygon labels
            # cannot define connected objects or the nearest solid boundary.
            if (manual!=255).all():
                ms=component_summary(manual==2); ps=component_summary(predictions[method]==2)
                out['bright_d50_error_px']=ps['d50']-ms['d50']
                manual_width=local_width(manual==0)
                pred_width=local_width(predictions[method]==0)
                for k in WIDTH_KEYS:
                    out[k+'_error_px']=pred_width[k]-manual_width[k]
            rows.append(out)
    table=pd.DataFrame(rows) if rows else pd.DataFrame(columns=['roi','site','split','status','method'])
    table.to_csv(OUT/'expert_evaluation.csv',index=False)
    write_json(OUT/'expert_evaluation_summary.json',dict(annotation_file_sha256=digest(path),
                manifest_id=manifest['manifest_id'],reviewed_measurable_rois=sum(a.get('review_status')=='reviewed' and a.get('measurable') is True for a in annotations['rois']),
                development_sites_scored=sorted(set(table.loc[(table.status=='expert_scored') & (table.split=='development'),'site'])) if len(table) else [],
                held_out_sites_scored=sorted(set(table.loc[(table.status=='expert_scored') & (table.split=='held_out_site'),'site'])) if len(table) else [],
                annotation_protocol='Partial masks: pixel metrics only. Full masks: component D50 and local-width errors. Methods compared at identical pixels/sites; crops never treated as independent material samples.'))
    print(f'Expert evaluation: {len(table)} rows; no unreviewed ROI scored.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',type=Path)
    p.add_argument('--render-only',action='store_true')
    args=p.parse_args()
    if args.annotations:
        evaluate_annotations(args.annotations)
    elif args.render_only:
        render_review(json.loads((OUT/'roi_manifest.json').read_text()))
    else:
        build()


if __name__=='__main__':
    main()
