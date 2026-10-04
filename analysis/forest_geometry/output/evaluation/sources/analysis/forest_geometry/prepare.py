"""Stage-bound E41S fixed geometry preparation from compatible caches/raw BSE.

Known: python -m analysis.forest_geometry.prepare --cohort known
Drop:  python -m analysis.forest_geometry.prepare --cohort drop --known-stage-receipt PATH
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/polaron-forest-geometry-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
import numpy as np
import pandas as pd

from polaron_qc import features
from analysis.forest_geometry import geometry as geo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'output'
ASSETS = Path(__file__).resolve().parent / 'assets'
EXAMPLES = (('Batch_2', '3806gxp0'), ('Batch_1', '4ih2ggld'),
            ('Batch_3', '71vgq3fw'), ('Batch_3', 'hzumfsms'))
CACHES = dict(shape='analysis/morphology/k_pilot/variants.csv',
    graph='analysis/ml_options/e_graph/variants.csv',
    width='analysis/morphology/benchmark/local_width_sites.csv',
    local='analysis/battery/output/neighbourhood_threshold_variants.csv')
MANIFESTS = dict(shape='analysis/morphology/k_pilot/manifest.json',
    graph='analysis/ml_options/e_graph/manifest.json',
    width='analysis/morphology/benchmark/roi_manifest.json',
    width_recovery='analysis/battery/output/void_manifest.json',
    local='analysis/battery/output/neighbourhood_manifest.json')
SOURCES = ['analysis/forest_geometry/geometry.py', 'analysis/forest_geometry/prepare.py',
    'analysis/forest_geometry/protocol_geometry.md', 'polaron_qc/features.py',
    'analysis/quality_policy_audit/policy.py', 'analysis/morphology/k_pilot/descriptors.py',
    'analysis/ml_options/e_graph/graph.py', 'analysis/morphology/benchmark_methods.py',
    'analysis/morphology/benchmark/provenance/benchmark_methods_extracted.py',
    'polaron_qc/battery_metrics.py',
    'analysis/battery/output/neighbourhood_measured_battery_metrics.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ast_function(path, name):
    tree = ast.parse(Path(path).read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(fn, include_attributes=False)


def _one(table, name, **filters):
    t = table
    for key, val in filters.items():
        t = t.loc[t[key].eq(val)]
    if len(t) != 1:
        raise ValueError(f'{name}: expected exactly one row for {filters}, found {len(t)}')
    return t.iloc[0]


def verify_bindings(tables):
    manifests = {k: json.loads((ROOT/p).read_text()) for k, p in MANIFESTS.items()}
    checks = {}
    # Current shape/graph definitions and raw-image binding survived earlier audits.
    essential = [('shape', 'polaron_qc/features.py'), ('shape', 'analysis/morphology/k_pilot/descriptors.py'),
                 ('graph', 'polaron_qc/features.py'), ('graph', 'analysis/ml_options/e_graph/graph.py')]
    for kind, path in essential:
        if sha(ROOT/path) != manifests[kind]['inputs'][path]:
            raise ValueError(f'Incompatible historical measurement module: {path}')
        checks[kind + ':' + path] = True
    m = manifests['width_recovery']['input_sha256']
    for path in (CACHES['width'], 'analysis/morphology/benchmark_methods.py'):
        if sha(ROOT/path) != m[path]:
            raise ValueError(f'Width recovery binding changed: {path}')
        checks['width:' + path] = True
    # Actual functions are identical; only experiment/ring wording changed.
    for fn in ('retained_voids', 'local_width'):
        if ast_function(ROOT/'analysis/morphology/benchmark_methods.py', fn) != ast_function(ROOT/'analysis/morphology/benchmark/provenance/benchmark_methods_extracted.py', fn):
            raise ValueError('Original width semantics changed')
        checks['width_ast:' + fn] = True
    if ast_function(ROOT/'polaron_qc/battery_metrics.py', 'measure_neighbourhoods') != ast_function(ROOT/'analysis/battery/output/neighbourhood_measured_battery_metrics.py', 'measure_neighbourhoods'):
        raise ValueError('Historical neighbourhood function changed')
    checks['window_source_function_AST_equal'] = True
    if sha(ROOT/'analysis/battery/output/neighbourhood_measured_battery_metrics.py') != manifests['local']['source_sha256_at_start']['polaron_qc/battery_metrics.py']:
        raise ValueError('Archived window-source hash differs')
    known = tables['shape'].loc[tables['shape']['mode'].eq('nominal')]
    if len(known) != 31 or known.duplicated(['batch', 'site']).any():
        raise ValueError('Known nominal shape cache is not31 unique crops')
    for row in known.itertuples():
        path = f'Dataset/{row.batch}/img_{row.site}_BSE.tif'
        digest = sha(ROOT/path)
        expected = [manifests['shape']['inputs'][path], manifests['graph']['inputs'][path],
                    manifests['width']['input_hashes'][path], manifests['width_recovery']['raw_BSE_sha256'][path],
                    manifests['local']['raw_bse_sha256'][path]]
        if any(digest != e for e in expected):
            raise ValueError(f'Raw input differs from a historical cache binding: {path}')
    checks['all_31_raw_BSE_hashes_match_five_historical_manifests'] = True
    return checks


def known_table(tables):
    """Cardinality-checked fixed4 variant assembly; no unfiltered merges."""
    shape, graph, width, local = [tables[k] for k in ('shape', 'graph', 'width', 'local')]
    rows = []
    pairs = pd.read_csv(ROOT/'analysis/morphology/k_pilot/pair_counts.csv')
    for variant, (offset, floor) in geo.VARIANTS.items():
        t = shape.loc[shape['mode'].eq(variant)]
        if len(t) != 31 or t.duplicated(['batch', 'site']).any():
            raise ValueError(f'Expected31 distinct shape rows in {variant}')
        for _, s in t.sort_values(['batch', 'site']).iterrows():
            filters = dict(batch=s.batch, site=s.site, threshold_offset=offset)
            g = _one(graph, 'graph', **filters, area_floor_px2=floor)
            w = _one(width, 'width', **filters)
            l = _one(local, 'local', **filters)
            if not np.allclose([s.th_lo, s.th_hi], [g.th_lo, g.th_hi], atol=1e-10, rtol=1e-10):
                raise ValueError('Historical threshold metadata disagreement')
            if bool(s.bright_low_contrast) != bool(g.bright_low_contrast) or int(s.bse_p1) != int(g.bse_p1):
                raise ValueError('Historical quality metadata disagreement')
            row = dict(batch=s.batch, site=s.site, cohort='known', variant=variant, version=geo.VERSION,
                bright_low_contrast=bool(s.bright_low_contrast), bse_p1=int(s.bse_p1),
                grey_pore=bool(int(s.bse_p1)>10), th_lo=s.th_lo, th_hi=s.th_hi,
                threshold_offset=offset, area_floor_px2=floor, trim_top=int(s.trim_top),
                trim_bottom=int(s.trim_bottom), H_trimmed=int(s.H), W=int(s.W))
            for k in geo.FEATURES:
                source = w if k in geo.PORE else g if k == geo.FEATURES[5] else l if k == geo.FEATURES[6] else s
                row[k] = float(source[k])
                reason = source.get(k + '_unavailable_reason', '')
                if k in geo.FEATURES[:3]: reason = s.shape_unavailable_reason
                if k in geo.PORE and not np.isfinite(source[k]): reason = 'no_retained_void_centreline_samples'
                if k == geo.FEATURES[6] and not np.isfinite(source[k]):
                    reason = 'fewer_than_2_complete_windows' if l.neighbourhood_windows_512px < 2 else 'no_bright_pixels_in_complete_windows'
                row[k + '_unavailable_reason'] = '' if pd.isna(reason) else str(reason)
            for source, keys in [(s, ['shape_objects_input', 'shape_objects_eligible', 'shape_objects_excluded', 'shape_circularity_above_one']),
                (g, ['graph_nodes', 'graph_edges', 'graph_components_available', 'graph_components_below_floor', 'graph_components_clipped', 'graph_zero_length_edges']),
                (w, ['n_retained_voids', 'n_width_samples', 'retained_void_area_frac', 'excluded_void_area_frac']),
                (l, ['neighbourhood_windows_512px', 'neighbourhood_window_coverage_512px', 'neighbourhood_windows_bright_present_512px'])]:
                for k in keys: row[k] = source[k]
            p = pairs.loc[pairs.batch.eq(s.batch)&pairs.site.eq(s.site)&pairs.threshold_offset.eq(offset)&pairs.kind.eq('cross')&pairs.lag_px.abs().eq(256)]
            if len(p) != 4: raise ValueError('Expected four signed cross-correlation pair domains')
            row.update(crosscorr_signed_domains=len(p), crosscorr_min_pair_pixels=int(p.pair_pixels.min()),
                crosscorr_min_source_phase_pixels=int(p.source_phase_pixels.min()),
                crosscorr_min_target_phase_pixels=int(p.target_phase_pixels.min()),
                crosscorr_min_source_complement_pixels=int((p.pair_pixels-p.source_phase_pixels).min()),
                crosscorr_min_target_complement_pixels=int((p.pair_pixels-p.target_phase_pixels).min()))
            rows.append(row)
    result = pd.DataFrame(rows)
    geo._validate_frame(result, with_variant=True)
    return result


def overlay(values, diagnostic, maps):
    raw = maps['raw_trimmed']; h, w = raw.shape
    height, width = min(768, h), min(1280, w)
    y, x = (h-height)//2, (w-width)//2
    sl = np.s_[y:y+height, x:x+width]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.7))
    for ax in axes[0]:
        ax.imshow(raw[sl], cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()
    axes[0,0].set_title('Fixed central raw BSE crop')
    rgba = np.zeros((height, width, 4))
    for mask, colour in ((maps['bright'], (1, .7, .1)), (maps['pore'], (.15, .72, .9))):
        use = mask[sl]; rgba[use, :3] = colour; rgba[use, 3] = .48
    axes[0,1].imshow(rgba)
    axes[0,1].set_title('Predicted bright gold / void cyan masks')
    sk = maps['width']['skeleton'][sl]
    # Display-only 3x3 maximum expansion prevents centreline pixels disappearing
    # when a1280px crop is rendered as a~400px report panel. Inputs never use it.
    display_width = ndi.maximum_filter(np.where(sk, 2*maps['width']['distance'][sl], 0.), size=3)
    widths = np.ma.masked_where(display_width == 0, display_width)
    im = axes[0,2].imshow(widths, cmap='cool', vmin=0, vmax=80, interpolation='nearest')
    fig.colorbar(im, ax=axes[0,2], label='2-D centreline width (px)', shrink=.7)
    axes[0,2].set_title(f"Retained void centreline · 3x3 display expansion\n{int(sk.sum()):,} crop / {diagnostic['n_width_samples']:,} full-field samples")
    ax = axes[1,0]; ax.imshow(raw[sl], cmap='gray', vmin=0, vmax=255)
    nodes, edges = maps['graph_nodes'], maps['graph_edges']
    points = nodes[['centroid-1', 'centroid-0']].to_numpy(float)
    if len(edges):
        segments = points[edges[['node_a', 'node_b']].to_numpy(int)] - [x, y]
        ax.add_collection(LineCollection(segments, colors='#22bfd9', linewidths=.6))
    ax.scatter(points[:,0]-x, points[:,1]-y, c='#ffbf41', s=8)
    ax.set(xlim=(0,width), ylim=(height,0), title='Full-site 3-NN graph, clipped for display'); ax.set_axis_off()
    ax=axes[1,1]; im=ax.imshow(maps['bright_window_fractions'], cmap='viridis', vmin=0, vmax=.2)
    fig.colorbar(im, ax=ax, label='Bright-mask fraction / 512-px window', shrink=.7)
    ax.set(title='Complete-window fractions (full field)', xlabel='Window x index', ylabel='Window y index')
    ax=axes[1,2]; p=maps['phase_pairs']; p=p.loc[p.kind.eq('cross')&p.lag_px.abs().eq(256)]
    labels=['x +256','x -256','y +256','y -256']
    vals=[p.loc[p.axis.eq(axis)&p.lag_px.eq(lag),'rho'].iloc[0] for axis,lag in [('x',256),('x',-256),('y',256),('y',-256)]]
    ax.bar(range(4), vals, color=['#2a78d6']*2+['#eb6834']*2)
    ax.set(xticks=range(4), xticklabels=labels, title='Signed finite-domain bright/void correlations', ylabel='Binary Pearson correlation', ylim=(-.2,.2));ax.axhline(0,color='gray',lw=.5)
    t=maps['shape_nodes']
    for ax, col, key, quantiles in zip(axes[2], ['shape_aspect','shape_circularity','solidity'], geo.FEATURES[:3], [(.25,.75),(.25,.75),(.1,)]):
        vals=t[col].to_numpy(float); ax.hist(vals, bins=24, color='#7998b5')
        for q in quantiles:
            if len(vals): ax.axvline(np.quantile(vals,q), color='#c93e4c', ls='--', label=f'Q{int(q*100)}')
        ax.set(title=f'{key}\nraw={values[key]:.4g}', xlabel=col, ylabel='Eligible components (coverage)');ax.legend(fontsize=8)
    flags=f"bright_low_contrast={values['bright_low_contrast']}; raised_black_level={values['grey_pore']}"
    fig.suptitle(f"{values['batch']}/{values['site']} · {flags}", fontsize=14)
    fig.text(.015,.012,'Unvalidated predicted masks. All8 values use the full trimmed field. Gold components/edges and centreline/window/pair counts are coverage, not additional samples.\nImage axes are not verified electrode axes. This fixed central crop is visual context, not a representative sampling claim.', fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.96))
    path=ASSETS/f"geometry_{values['batch']}_{values['site']}.png"
    fig.savefig(path,dpi=105);plt.close(fig)
    return dict(batch=values['batch'],site=values['site'],path=str(path.relative_to(ROOT)),sha256=sha(path),
        box_trimmed_yxyx=[int(y),int(x),int(y+height),int(x+width)],
        box_raw_yxyx=[int(y+values['trim_top']),int(x),int(y+height+values['trim_top']),int(x+width)],
        selection='Fixed central min(768,H)x min(1280,W), full-frame measurements',
        expert_validation='none; unvalidated predicted masks',
        displayed_width_centreline_samples=int(sk.sum()),
        full_field_width_centreline_samples=int(diagnostic['n_width_samples']),
        width_display='3x3 maximum expansion only for display; scalar measurements unchanged')


def _snapshot(label, paths):
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    texts={p:(ROOT/p).read_text() for p in SOURCES}
    receipt=dict(experiment='E41S',version=geo.VERSION,cohort=label,input_hashes=hashes,
        source_text=texts,features=geo.FEATURES,definitions=geo.DEFINITIONS,
        statistical_unit='crop; parent image dependence unresolved; counts are coverage only')
    write_json(OUT/f'geometry_{label}_input_receipt.json',receipt)
    return hashes


def _save_table(table, label):
    table.to_csv(OUT/f'geometry_{label}_variants.csv', index=False)
    nom=geo.nominal(table)
    nom.to_csv(OUT/f'geometry_{label}.csv',index=False)
    coverage=[]
    for variant in geo.VARIANTS:
        t=table.loc[table.variant.eq(variant)].copy()
        if t.empty: continue
        X,names,audit,q=geo.prepare(t)
        audit['variant']=variant;coverage.append(audit)
        gated=t.copy();gated[names]=X
        gated.to_csv(OUT/f'geometry_{label}_{variant}_gated.csv',index=False)
    pd.concat(coverage,ignore_index=True).to_csv(OUT/f'geometry_{label}_availability.csv',index=False)
    return nom


def prepare_known():
    start=time.perf_counter()
    tables={k:pd.read_csv(ROOT/p) for k,p in CACHES.items()}
    paths=[ROOT/p for p in list(CACHES.values())+list(MANIFESTS.values())+SOURCES+
           ['analysis/morphology/k_pilot/pair_counts.csv']]
    nominal=tables['shape'].loc[tables['shape']['mode'].eq('nominal')]
    paths += [ROOT/'Dataset'/r.batch/f'img_{r.site}_BSE.tif' for r in nominal.itertuples()]
    before=_snapshot('known',paths)
    binding=verify_bindings(tables)
    table=known_table(tables)
    nom=_save_table(table,'known')
    # The model runner can fit known-only models while replay verification runs;
    # these outputs are experimental until the verification receipt passes.
    parity=[];examples=[]
    for batch,site in EXAMPLES:
        row=_one(nom,'nominal replay',batch=batch,site=site)
        raw=features.load_image(ROOT/'Dataset'/batch,site,'BSE')
        values,diag,maps=geo.extract_site(raw,metadata=dict(batch=batch,site=site,cohort='known',variant='nominal'),
            thresholds=(row.th_lo,row.th_hi),return_maps=True)
        for key in geo.FEATURES:
            match=bool(np.isclose(values[key],row[key],atol=1e-9,rtol=1e-9,equal_nan=True))
            parity.append(dict(batch=batch,site=site,feature=key,cached=float(row[key]),fresh=float(values[key]),match=match))
        for key in ['bright_low_contrast','bse_p1','th_lo','th_hi','trim_top','trim_bottom']:
            if not np.isclose(float(values[key]),float(row[key]),atol=1e-9):
                raise ValueError(f'Raw replay metadata differs: {site}/{key}')
        examples.append(overlay(values,diag,maps))
        print(f'Fresh8 parity + visual: {batch}/{site}',flush=True)
    parity_frame=pd.DataFrame(parity);parity_frame.to_csv(OUT/'geometry_nominal_replay.csv',index=False)
    if not parity_frame.match.all(): raise ValueError('Fresh geometry replay differs from warm nominal cache')
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    if changed: raise ValueError(f'Protected geometry inputs changed during extraction: {changed}')
    write_json(OUT/'geometry_examples.json',dict(experiment='E41S',examples=examples))
    write_json(OUT/'geometry_known_verification.json',dict(experiment='E41S',version=geo.VERSION,
        known_sites=len(nom),variant_rows=len(table),fixed8=geo.FEATURES,
        cache_reuse='Exact compatible E24/E25/E26G/E30K definitions; receipts bind current reused tables and historical source/raw provenance',
        binding_checks=binding,fresh_nominal_example_cells=len(parity),fresh_nominal_matches=int(parity_frame.match.sum()),
        inputs_unchanged=True,known_outputs_ready=True,elapsed_seconds=time.perf_counter()-start,
        output_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('geometry_known*.csv')}))
    print('Geometry known stage verified:31 crops/124variants/32 fresh nominal cells.',flush=True)


def require_known_stage(path):
    if path is None: raise ValueError('Drop extraction requires explicit known-only stage receipt')
    p=Path(path)
    if not p.is_file(): raise ValueError('Known-only stage receipt is missing')
    receipt=json.loads(p.read_text())
    if receipt.get('stage')!='known_only_complete':
        raise ValueError('Known-only models/metrics have not been saved')
    return dict(path=str(p.resolve()),sha256=sha(p),stage=receipt['stage'])


def prepare_drop(stage_path):
    stage=require_known_stage(stage_path) # checked before ANY drop directory/raw read
    raw_root=ROOT/'Hackathon-Polaron-test'
    files=sorted(raw_root.glob('img_*_BSE.tif'))
    if len(files)!=3: raise ValueError('Expected3 organiser-drop BSE crops')
    before=_snapshot('first_drop',files+[ROOT/p for p in SOURCES])
    records=[]
    for p in files:
        site=p.stem[len('img_'):-len('_BSE')]
        raw=features.load_image(raw_root,site,'BSE')
        values,diag=geo.extract_site(raw,metadata=dict(batch=raw_root.name,site=site,
            cohort='first_drop_development',variant='nominal'))
        records.append(values|diag)
        print(f'Raw-only drop geometry: {site}',flush=True)
    table=pd.DataFrame(records)
    _save_table(table,'first_drop')
    known=pd.read_csv(OUT/'geometry_known.csv')
    pd.concat([known,table],ignore_index=True).to_csv(OUT/'geometry_development_raw_34.csv',index=False)
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    if changed: raise ValueError(f'Drop extraction source/raw inputs changed: {changed}')
    write_json(OUT/'geometry_first_drop_verification.json',dict(experiment='E41S',version=geo.VERSION,
        stage_receipt=stage,sites=len(table),inputs_unchanged=True,
        labels_read=False,outcomes_read=False,scope='Post-feedback development features; not new validation',
        output_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('geometry_first_drop*.csv')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',choices=['known','drop'],default='known')
    parser.add_argument('--known-stage-receipt',type=Path)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    if args.cohort=='known':prepare_known()
    else:prepare_drop(args.known_stage_receipt)


if __name__=='__main__':main()
