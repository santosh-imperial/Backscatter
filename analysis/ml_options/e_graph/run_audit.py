"""Task E / E26G: fixed known-site graph audit; production QC stays frozen."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.ml_options.e_graph.graph import KEYS, DEFINITIONS, build_graph, usable
from analysis.morphology.run_analysis import component_table
from polaron_qc import features, PRIMARY_KPIS, MATERIAL_KPIS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 20261003
N_BOOT = 4000
ACQ = ('bright_sep', 'bse_p1', 'bse_std', 'bse_empty_bin_frac', 'etd_boundary_sharpness', 'H')
BASE = ('bright_frac', 'bright_d50', 'local_bright_std_512px', 'local_bright_std_1024px',
        'graph_node_density_per_mpx', 'graph_node_log_area_median', 'graph_node_log_area_iqr')
VIEWS = ('bright_usable', 'quality_matched', 'ordinary_reference')
PAIRS = (('Batch_3', 'Batch_1'), ('Batch_3', 'Batch_2'), ('Batch_1', 'Batch_2'))
EXAMPLES = (('Batch_2', '3806gxp0'), ('Batch_3', 'hzumfsms'),
            ('Batch_1', '4ih2ggld'), ('Batch_3', '71vgq3fw'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def bootstrap_difference(a, b, seed):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(len(a), len(b)) < 2: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    am = np.median(a[rng.integers(len(a), size=(N_BOOT, len(a)))], axis=1)
    bm = np.median(b[rng.integers(len(b), size=(N_BOOT, len(b)))], axis=1)
    return np.quantile(bm-am, [.025, .975])


def comparisons(variants):
    rows = []
    for floor in (50, 100):
        for offset in (-5, 0, 5):
            t = variants[(variants.area_floor_px2 == floor) & (variants.threshold_offset == offset)]
            for view in VIEWS:
                for key in KEYS:
                    good = t.loc[usable(t, key, view)]
                    for reference, batch in PAIRS:
                        a, b = good[good.batch.eq(reference)][key], good[good.batch.eq(batch)][key]
                        lo, hi = bootstrap_difference(a, b, SEED + len(rows))
                        delta = b.median()-a.median()
                        deleted = []
                        if min(len(a), len(b)) >= 2:
                            deleted = [b.median()-a.drop(i).median() for i in a.index]
                            deleted += [b.drop(i).median()-a.median() for i in b.index]
                        rows.append(dict(view=view, kpi=key, area_floor_px2=floor,
                            threshold_offset=offset, reference=reference, batch=batch,
                            n_reference=len(a), n_batch=len(b), median_reference=a.median(), median_batch=b.median(),
                            median_difference=delta, ci_low=lo, ci_high=hi,
                            deletion_min=min(deleted) if deleted else np.nan,
                            deletion_max=max(deleted) if deleted else np.nan))
    return pd.DataFrame(rows)


def controls(sites):
    correlations, predictions = [], []
    for key in KEYS:
        t = sites.loc[usable(sites, key)].copy()
        groups = [('pooled', t)] + [(b, t[t.batch.eq(b)]) for b in sorted(t.batch.unique())]
        groups += [('Batch_3_ordinary', sites.loc[usable(sites, key, 'ordinary_reference') & sites.batch.eq('Batch_3')])]
        for group, g in groups:
            for kind, covs in (('acquisition', ACQ), ('node_loading_baseline', BASE)):
                for cov in covs:
                    f = g[[key, cov]].replace([np.inf, -np.inf], np.nan).dropna()
                    rho = float(spearmanr(f[key], f[cov]).statistic) if len(f) >= 4 and min(f[key].nunique(), f[cov].nunique()) > 1 else np.nan
                    correlations.append(dict(kpi=key, group=group, control=kind, covariate=cov, n_sites=len(f), rho=rho))
        for kind, covs in (('acquisition', ACQ), ('node_loading_baseline', BASE)):
            x = t[list(covs)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            y = t[key].to_numpy(float)
            pred = np.full(len(y), np.nan)
            if len(y) >= 5 and np.ptp(y) > 0:
                for i in range(len(y)):
                    train = np.arange(len(y)) != i
                    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=1))
                    model.fit(x[train], y[train]); pred[i] = model.predict(x[i:i+1])[0]
            finite = np.isfinite(pred)
            r2 = 1-np.sum((y[finite]-pred[finite])**2)/np.sum((y[finite]-y[finite].mean())**2) if finite.sum() >= 5 and np.ptp(y[finite]) else np.nan
            for i, r in enumerate(t.itertuples()):
                predictions.append(dict(kpi=key, control=kind, batch=r.batch, site=r.site,
                    observed=y[i], prediction=pred[i], loo_r2=r2, n_sites=len(y), covariates=';'.join(covs)))
    return pd.DataFrame(correlations), pd.DataFrame(predictions)


def overlay(a, top, batch, site, tables, components):
    h, w = a.shape
    height, width = min(768, h), min(1280, w)
    y0, x0 = (h-height)//2, (w-width)//2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax in axes:
        ax.imshow(a, cmap='gray', vmin=0, vmax=255)
        ax.set_xlim(x0, x0+width); ax.set_ylim(y0+height, y0)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title('Fixed central raw BSE crop')
    for ax, floor in zip(axes[1:], (50, 100)):
        metrics, nodes, edges = tables[floor]
        points = nodes[['centroid-1', 'centroid-0']].to_numpy(float)
        if len(edges):
            segments = points[edges[['node_a', 'node_b']].to_numpy(int)]
            ax.add_collection(LineCollection(segments, colors='#49d7ed', linewidths=.65, alpha=.75))
        ax.scatter(points[:, 0], points[:, 1], s=10, color='#ffc65b')
        clipped = components[(components.area >= floor) & components.touches_edge]
        ax.scatter(clipped['centroid-1'], clipped['centroid-0'], s=30, color='#fa5585', marker='x')
        in_crop = nodes[(nodes['centroid-0'] >= y0) & (nodes['centroid-0'] < y0+height) &
                        (nodes['centroid-1'] >= x0) & (nodes['centroid-1'] < x0+width)]
        for r in in_crop.head(30).itertuples():
            # Column names containing hyphens are not stable namedtuple fields.
            row = nodes.loc[r.Index]
            ax.text(row['centroid-1'], row['centroid-0'], str(int(row.label)), color='white', fontsize=5)
        ax.set_title(f'Area floor {floor} px² · full-site {metrics["graph_nodes"]} nodes / {metrics["graph_edges"]} edges')
    fig.suptitle(f'{batch}/{site} · raw crop x={x0}:{x0+width}, y={top+y0}:{top+y0+height}', fontsize=11)
    fig.text(.01, .01, 'Gold: retained centroid; cyan: fixed 3-NN proximity edge; pink ×: excluded image-clipped component.\nGraphs are built on the entire trimmed site, then cropped for display. Labels identify mask components, not chemically verified particles.', fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.91))
    name = f'overlay_{batch}_{site}.png'
    fig.savefig(OUT/name, dpi=150); plt.close(fig)
    return dict(batch=batch, site=site, image=name, x=x0, y_trimmed=y0, y_raw=top+y0, width=width, height=height)


def report(sites, comp, predictions, correlations, crops, manifest):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ('#2a78d6', '#eb6834', '#1baf7a')
    for key, ax in zip(KEYS, axes.ravel()):
        for i, (batch, color) in enumerate(zip(sorted(sites.batch.unique()), colors)):
            t = sites[sites.batch.eq(batch)]
            ok = usable(t, key)
            ax.scatter(np.full(ok.sum(), i), t.loc[ok, key], color=color, label=batch)
            ax.scatter(np.full((~ok).sum(), i), t.loc[~ok, key], color='#e34948', marker='x')
            if ok.any(): ax.plot([i-.15, i+.15], [t.loc[ok, key].median()]*2, color='black')
        ax.set_xticks(range(3), ['B1','B2','B3'])
        ax.set_ylabel(DEFINITIONS[key][0]); ax.set_title(key.replace('graph_', '').replace('_',' '))
        ax.legend(fontsize=7)
    fig.suptitle('Full-site descriptors · red × excluded low contrast · black line median')
    fig.tight_layout(); fig.savefig(OUT/'site_descriptors.png', dpi=150); plt.close(fig)
    nominal = comp[(comp.area_floor_px2 == 50) & comp.threshold_offset.eq(0)]
    lines = ['# Fixed bright-object graph audit (E26G / task E)', '',
        'Known-batch development. Proximity edges are geometric constructions, not electrical or physical contacts. Expert masks and specimen independence remain unconfirmed. No primary KPI, classifier or verdict input changes.', '',
        'Three nearest distinct centroid neighbours, undirected union; 8-connected bright masks after existing smoothing/opening; component floor 50 px², clipped objects excluded. Sensitivity repeats all sites at thresholds −5/0/+5 and area floors 50/100 px². These settings were fixed before new outputs. At least 10 nodes/10 edges is a pilot coverage gate, not additional statistical n.', '',
        'Bright-usable excludes observed low contrast; quality-matched additionally excludes grey-pore preparation flags. Ordinary-reference sensitivity additionally removes known long-void reference sites, without declaring a clean baseline. Grey-pore is not required for a bright-only measurand. Missing flags abstain; unresolved pore-mode diagnostics remain disclosed.', '',
        'Intervals are 4000-resample percentile site-bootstrap intervals, conditional on provisional site independence. Threshold/object-floor sensitivity and deletion ranges are separate from sampling uncertainty. No p-values, classifier AUC or manufacturing defect accuracy were estimated.', '',
        '## Bright-usable nominal comparisons', '',
        '| descriptor | incoming − reference | n incoming / reference | median difference | site-bootstrap 95% interval |',
        '|---|---|---|---|---|']
    for r in nominal[nominal.view.eq('bright_usable')].itertuples():
        lines.append(f'| `{r.kpi}` | {r.batch} − {r.reference} | {r.n_batch}/{r.n_reference} | {r.median_difference:.5g} | [{r.ci_low:.5g}, {r.ci_high:.5g}] |')
    lines += ['', '## Sensitivity and baseline controls', '',
        'All pairwise views and six threshold/floor variants are saved in comparisons.csv. Acquisition and node/loading-only ridge controls use alpha=1 with imputation and scaling refitted inside each held-out-site fold. Within-batch correlations accompany pooled results. Positive predictability suggests redundancy/confounding; poor prediction does not establish novelty or acquisition invariance.', '',
        '| descriptor | acquisition LOO R² | node/loading LOO R² | strongest within-batch acquisition correlation |', '|---|---|---|---|']
    for key in KEYS:
        p = predictions[predictions.kpi.eq(key)]
        ac = correlations[correlations.kpi.eq(key) & correlations.control.eq('acquisition') &
                          correlations.group.isin(['Batch_1','Batch_2','Batch_3'])].dropna(subset=['rho'])
        strongest = 'unavailable'
        if len(ac):
            r = ac.loc[ac.rho.abs().idxmax()]; strongest = f'{r.group}/{r.covariate}: rho={r.rho:.3f}, n={r.n_sites}'
        def r2(kind):
            v = p.loc[p.control.eq(kind), 'loo_r2']; return f'{v.iloc[0]:.3f}' if len(v) else 'unavailable'
        lines.append(f'| `{key}` | {r2("acquisition")} | {r2("node_loading_baseline")} | {strongest} |')
    lines += ['', '## Retain/defer decision', '',
        'Retain the fixed graph as an exploratory image-review tool. Defer all four descriptors as material/QC drivers until independent component review, sensitivity and specimen/process validation. No GNN, learned defect classifier or battery-mechanism advantage is claimed. Finite-frame neighbour bias remains despite removing clipped objects; object counts and edges do not increase n.', '',
        'The four overlay sites were prespecified, with crops fixed at frame centre rather than chosen for separation. Full-frame graphs are cropped only for display; values come from full-site tables. Both clipping and sub-floor exclusions are recorded in coverage fields. Node/edge source tables include exact coordinates and component labels.', '',
        'Pre-presentation checks: C01–C06/C09/C13/C14/C16/C17/C21/C22/C28–C30. Syntax and graph tests validate conventions, not expert segmentation accuracy or unseen-batch generalisation.', '']
    (OUT/'findings.md').write_text('\n'.join(lines))
    def img(name):
        data = base64.b64encode((OUT/name).read_bytes()).decode()
        return f'<img style="width:100%;height:auto" src="data:image/png;base64,{data}">'
    tables = nominal.to_html(index=False, float_format=lambda x: f'{x:.5g}', escape=True)
    defs = ''.join(f'<li><strong>{html.escape(k)}</strong> ({html.escape(u)}): {html.escape(d)}</li>' for k,(u,d) in DEFINITIONS.items())
    images = ''.join(f'<details open><summary>{html.escape(c["batch"]+"/"+c["site"])}</summary>{img(c["image"])}</details>' for c in crops)
    page = '<!doctype html><html><head><meta charset="utf-8"><title>Fixed particle graph audit</title><style>body{font:16px system-ui;margin:30px auto;max-width:1200px;line-height:1.5;background:#f6f8fb;color:#182338}table{font-size:12px;border-collapse:collapse}td,th{padding:5px;border:1px solid #ccd6df}details{margin:20px 0;padding:15px;background:white}.scroll{overflow:auto}</style></head><body>'
    page += '<h1>Fixed particle-neighbourhood graph audit</h1><p>Exploratory known-site geometry · expert validation pending · excluded from QC verdicts.</p><ul>'+defs+'</ul>'
    page += '<h2>Full-site measurements</h2>'+img('site_descriptors.png')+'<h2>Image evidence and object-floor sensitivity</h2>'+images
    page += '<h2>Nominal comparisons and sampling intervals</h2><div class="scroll">'+tables+'</div><h2>Review findings</h2><pre style="white-space:pre-wrap">'+html.escape('\n'.join(lines))+'</pre></body></html>'
    (OUT/'report.html').write_text(page)


def main():
    start = time.perf_counter()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert protocol['keys'] == list(KEYS) and protocol['n_boot'] == N_BOOT
    cache = ROOT/'analysis/battery/output/neighbourhood_sites.csv'
    metadata = pd.read_csv(cache).sort_values(['batch','site'])
    inputs = [cache, OUT/'protocol.json', OUT/'graph.py', Path(__file__),
              ROOT/'polaron_qc/features.py', ROOT/'analysis/morphology/run_analysis.py']
    inputs += [ROOT/'Dataset'/r.batch/f'img_{r.site}_BSE.tif' for r in metadata.itertuples()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    records, node_tables, edge_tables, crops = [], [], [], []
    for row in metadata.itertuples():
        a = features.load_image(ROOT/'Dataset'/row.batch, row.site, 'BSE')
        top, bottom = features.bright_bands(a)
        a = a[top:len(a)-bottom if bottom else None]
        sm = features.smooth_bse(a)
        thresholds = features.phase_thresholds(sm)
        assert np.allclose([thresholds['th_lo'], thresholds['th_hi']], [row.th_lo, row.th_hi])
        assert bool(thresholds['bright_low_contrast']) == bool(row.bright_low_contrast)
        example_tables = {}; example_components = None
        for offset in (-5,0,5):
            mask = ndi.binary_opening(sm > row.th_hi+offset, iterations=1)
            ct = component_table(mask, 50)
            for floor in (50,100):
                m, nodes, edges = build_graph(ct, floor)
                header = dict(batch=row.batch, site=row.site, threshold_offset=offset, area_floor_px2=floor)
                m['graph_node_density_per_mpx'] = len(nodes)/a.size*1e6
                record = row._asdict() | header | m
                record.update(trim_top=top, trim_bottom=bottom)
                records.append(record)
                nodes = nodes.assign(**header, node_id=np.arange(len(nodes)))
                edges = edges.assign(**header)
                node_tables.append(nodes); edge_tables.append(edges)
                if offset == 0 and (row.batch,row.site) in EXAMPLES:
                    example_tables[floor] = (m,nodes,edges); example_components = ct
        if example_tables: crops.append(overlay(a,top,row.batch,row.site,example_tables,example_components))
        print(f'{row.batch}/{row.site}: six fixed graph variants', flush=True)
    variants = pd.DataFrame(records)
    sites = variants[variants.threshold_offset.eq(0) & variants.area_floor_px2.eq(50)].copy()
    for key in KEYS:
        grouped = variants.groupby(['batch','site'])[key]
        endpoints = grouped.agg(['min','max','count'])
        for agg in ('min','max','count'):
            sites[key+'_sensitivity_'+agg] = [endpoints.loc[(r.batch,r.site),agg] for r in sites.itertuples()]
    comp = comparisons(variants)
    correlations, predictions = controls(sites)
    variants.to_csv(OUT/'variants.csv',index=False)
    sites.to_csv(OUT/'sites.csv',index=False)
    pd.concat(node_tables,ignore_index=True).to_csv(OUT/'nodes.csv.gz',index=False,compression='gzip')
    pd.concat(edge_tables,ignore_index=True).to_csv(OUT/'edges.csv.gz',index=False,compression='gzip')
    comp.to_csv(OUT/'comparisons.csv',index=False)
    correlations.to_csv(OUT/'correlations.csv',index=False)
    predictions.to_csv(OUT/'control_predictions.csv',index=False)
    changed = [p for p,d in before.items() if sha(ROOT/p) != d]
    if changed: raise RuntimeError(f'Inputs changed during run: {changed}')
    manifest = dict(experiment='E26G',task='E',version=protocol['version'],date='2026-10-03',
        inputs=before,inputs_unchanged=True,raw_data_root=str((ROOT/'Dataset').resolve()),
        n_sites=len(sites),n_variants=len(variants),n_boot=N_BOOT,seed=SEED,
        primary_kpis=list(PRIMARY_KPIS),material_kpis=list(MATERIAL_KPIS),
        elapsed_seconds=time.perf_counter()-start,crops=crops,
        provenance='Fresh raw BSE mask reconstruction; historical verified metadata reuse; known-data development, no unseen evaluation.')
    for source in (OUT/'graph.py',Path(__file__),ROOT/'polaron_qc/features.py',ROOT/'analysis/morphology/run_analysis.py'):
        dest=OUT/'provenance'/source.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(source.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(sites,comp,predictions,correlations,crops,manifest)
    print(f'Complete: {len(sites)} sites; {len(variants)} variants; {len(comp)} descriptive comparisons.',flush=True)


if __name__ == '__main__': main()
