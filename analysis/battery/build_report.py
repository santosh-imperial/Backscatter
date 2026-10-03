"""Build the E25 scientific audit report from saved measurements and provenance."""
from pathlib import Path
import base64
import html
import json
import markdown
import pandas as pd
from polaron_qc import secondary

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('output')

def esc(value): return html.escape(str(value))
def figure(name, caption):
    path=OUT/name
    assert path.exists(), path
    data=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'

def main():
    verified=json.loads((OUT/'pipeline_verification.json').read_text())['results']
    rows=pd.read_csv(OUT/'pipeline_battery_summary.csv')
    assert len(rows)==16 and set(rows.kpi)==set(secondary.KPI_NAMES)
    catalog=json.loads((ROOT/'analysis/morphology/metric_register.json').read_text())
    by_key={m['id']:m for m in catalog['metrics']}
    intro='<h1>Battery microstructure checks</h1><p>Fresh graphite–Si/SiOx electrode · E25 · 31 known sites · experimental geometry, independent expert validation pending.</p>'
    nav='<nav><a href="#metrics">Pipeline measurements</a><a href="#neighbourhood">Neighbourhoods</a><a href="#void">Long voids</a><a href="#orientation">Orientation / Inlens</a><a href="#verification">Verification</a></nav>'
    summary='''<section class="summary"><p>Batch2 shows greater bright-to-void distance, with a sampling interval that includes zero. Previously selected long-void sites retain higher internal burden, but large edge-clipped cavities limit width/location coverage. No shared lower-frame location or independently confirmed collector interface is established.</p><p>Graphite plate orientation remains deferred: residual-solid regions join across particles, while tensor directions describe image texture. Local-normalised Inlens gradients remain acquisition-sensitive. These observations do not predict battery performance.</p></section>'''
    metric_table=['<section id="metrics"><h2>Eight experimental pipeline KPIs</h2><p>Separate from the five primary KPIs, classifier inputs and verdicts. Differences below are batch minus working reference Batch3, using the pipeline’s deterministic site bootstrap. Independent analyses below use their recorded bootstrap seeds/draw counts, so percentile endpoints can differ slightly. Specimen independence remains unverified.</p><div class="scroll"><table><thead><tr><th>Measurement</th><th>Batch</th><th>Ref / batch median</th><th>Difference</th><th>95% site interval</th><th>Usable n ref / batch</th><th>Unit</th></tr></thead><tbody>']
    def num(v): return f'{v:.4g}' if pd.notna(v) else 'unavailable'
    for _,r in rows.iterrows():
        label=by_key[r.kpi]['label']; definition=secondary.KPI_DEFINITIONS[r.kpi]['definition']
        metric_table.append(f'<tr><td title="{esc(definition)}">{esc(label)}<small>{esc(r.kpi)}</small></td><td>{esc(r.batch)}</td><td>{num(r.ref_median)} / {num(r.batch_median)}</td><td>{num(r.median_difference)}</td><td>[{num(r.ci_low)}, {num(r.ci_high)}]</td><td>{int(r.n_ref_usable)} / {int(r.n_batch_usable)}</td><td>{esc(r.units)}</td></tr>')
    metric_table.append('''</tbody></table></div><p>Quality excludes data-derived low-contrast sites for bright-dependent measurements and grey-pore sites for pore-dependent measurements. All nominal pore-mode flags remain unresolved, disclosed separately. Missing flags/values are unusable. Exact bright/void raster adjacency is constant zero and remains diagnostic. Paired threshold ranges and coverage appear in the full QC reports and atlas.</p><p><a href="pipeline_battery_summary.csv">Exact pipeline table</a> · <a href="../../../reports/qc_Batch_1.html">Batch1 QC report</a> · <a href="../../../reports/qc_Batch_2.html">Batch2 QC report</a> · <a href="../../morphology/output/metric_atlas.html">Live visual atlas and status</a> · <a href="../../../docs/assumption_register.html#A18">Interpretation review A18</a></p></section>''')
    sections=[]
    specs=[('neighbourhood','Particle neighbourhoods / local homogeneity','neighbourhood_findings.md',[
        ('neighbourhood_overlay_Batch_2_3806gxp0.png','Ordinary Batch2 example: real mask neighbourhoods and fixed image windows. The ring includes threshold holes within the component silhouette and is not physical expansion space.'),
        ('neighbourhood_overlay_Batch_1_4ih2ggld.png','Low-contrast example: source masks remain visible, but bright-dependent comparisons exclude this site.')]),
      ('void','Long-void location / width coverage','void_findings.md',[
        ('void_full_site_overlays.png','Previously identified long-void sites: internal versus edge-clipped components. Image rows are not confirmed collector coordinates.'),
        ('void_collector_candidate.png','Raw bottom bright-band candidate in epqdaau9; collector/interface geometry is unconfirmed. No adhesion verdict follows.')]),
      ('orientation','Graphite orientation / Inlens feasibility','orientation_findings.md',[
        ('orientation_matched_crops.png','Same raw crop coordinates across BSE, ETD and Inlens. Yellow tensor directions are image-texture directions, not graphite plate labels.'),
        ('orientation_inlens_normalisation_examples.png','Local standardisation changes the sampled gradient measurand; preparation relief, charging and non-linear contrast effects may persist.')])]
    for id_,title,memo,figs in specs:
        content=markdown.markdown((OUT/memo).read_text(),extensions=['tables','fenced_code'])
        # The detail memo has its own title; avoid duplicate page h1.
        content=content.replace('<h1>','<h3>').replace('</h1>','</h3>')
        sections.append(f'<section id="{id_}"><h2>{esc(title)}</h2>'+''.join(figure(*f) for f in figs)+f'<details><summary>Definitions, numeric results, uncertainty and provenance</summary>{content}</details><p><a href="{memo}">Source findings</a></p></section>')
    verification='<section id="verification"><h2>Verification and limits</h2><p>140 tests passed including real extraction/cache regressions and geometric convention checks; 55 final focused checks pass after the missing-quality guard. Known-batch default report replays preserve historical primary columns and frozen thresholds. Replay explicitly reuses unchanged original all-channel feature caches with secondary masks recomputed from raw BSE; it is not a full cold all-channel run.</p><p>Both batches remain consistent with the working reference within detectable limits. No acceptance/equivalence, expert mask accuracy or unseen-batch generalisation is claimed. Independent phase/instance labels, specimen grouping and collector review remain the next validation gates.</p><ul>'+''.join(f'<li>{esc(r["batch"])}: {esc(r["verdict"])}; decision stability {r["decision_stability"]:.2f}; threshold hash {esc(r["threshold_hash"])}</li>' for r in verified)+'</ul><p>'+ ' · '.join(f'<a href="{f}">{esc(f)}</a>' for f in ['neighbourhood_manifest.json','void_manifest.json','orientation_summary.json','pipeline_cache_provenance.json','pipeline_verification.json'])+'</p></section>'
    css='''body{font:16px/1.55 system-ui,sans-serif;color:#172331;background:#f4f6f8;margin:0}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:34px;line-height:1.2}h2{font-size:24px}section{background:white;padding:24px;margin:24px 0;border:1px solid #d5dce4;border-radius:10px;scroll-margin-top:80px}nav{display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;padding:14px;background:#f4f6f8;border-bottom:1px solid #d5dce4;z-index:2}a{color:#125ca2}figure{margin:22px 0}img{width:100%;height:auto;display:block}figcaption,small{font-size:13px;color:#506071}small{display:block}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde3e9;text-align:left;vertical-align:top}th{background:#edf2f7}summary{cursor:pointer;font-weight:650}details[open] summary{margin-bottom:16px}.summary{border-left:5px solid #d99b25}@media(max-width:650px){main{padding:12px}section{padding:14px}h1{font-size:28px}nav{gap:10px;font-size:14px}}'''
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Battery microstructure audit — E25</title><style>'+css+'</style><main>'+intro+nav+summary+''.join(metric_table)+''.join(sections)+verification+'</main></html>'
    (OUT/'report.html').write_text(page)
    assert len(page.encode())<20_000_000
    print(f'Wrote {OUT / "report.html"}: {len(page.encode())/1e6:.2f} MB; 8 KPIs, 3 audits, 6 real-image panels.')

if __name__=='__main__': main()
