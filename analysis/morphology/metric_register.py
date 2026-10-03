"""Validate/render the live metric inventory; preserve manual reviews and history.

Edit metric_register.json to add metrics or record expert reviews, then run:
    python -m analysis.morphology.metric_register
After a completed benchmark build:
    python -m analysis.morphology.metric_register --sync-benchmark
No automated operation promotes a metric into the frozen primary family.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from datetime import date
from pathlib import Path

import pandas as pd
from polaron_qc import PRIMARY_KPIS
from analysis.morphology.run_analysis import MORPH

ROOT=Path(__file__).resolve().parents[2]
REGISTER=Path(__file__).with_name('metric_register.json')
REQUIRED={'id','label','family','keys','channels','units','definition','interpretation',
          'implementation_status','evidence_status','qc_role','limitations','evidence_paths',
          'experiments','visual_kind','review_status','review_notes','history'}


def validate(data):
    assert data['schema_version']==1
    metrics=data['metrics']; ids=[m['id'] for m in metrics]
    assert len(ids)==len(set(ids)), 'Metric IDs must be unique'
    assert set(MORPH)<=set(ids), 'Every E18 descriptor must be tracked'
    assert set(m['id'] for m in metrics if m['qc_role']=='primary')==set(PRIMARY_KPIS)
    for m in metrics:
        assert REQUIRED<=set(m), (m['id'],'missing fields')
        assert m['keys'] and m['definition'] and m['limitations']
        assert m['review_status'] in ('unreviewed','confirmed','refuted','needs_revision')
        assert m['history'] and m['history'][-1]['status']==m['implementation_status']
        for path in m['evidence_paths']:
            if m['implementation_status'] not in ('in_progress','proposed','deferred'):
                assert (ROOT/path).exists(), (m['id'],path,'missing evidence')
    assert len({m['id'] for m in data['methods']})==len(data['methods'])
    checks=data.get('battery_checks',[])
    check_ids={c['id'] for c in checks}
    assert len(check_ids)==len(checks), 'Battery check IDs must be unique'
    for c in checks:
        assert set(c['metric_ids'])<=set(ids), (c['id'],'unregistered metric')
        assert c['qc_role']=='hypothesis_only' and c['validation_needed']
    for m in metrics:
        assert set(m.get('battery_check_ids',[]))<=check_ids


def update_status(entry,status,reason,evidence=None):
    if entry['implementation_status']!=status:
        entry.setdefault('history',[]).append(dict(date=date.today().isoformat(),status=status,reason=reason))
        entry['implementation_status']=status
    if evidence is not None:
        entry['evidence_status']=evidence


def sync_benchmark(data):
    out=REGISTER.parent/'benchmark'
    manifest=json.loads((out/'roi_manifest.json').read_text())
    assert manifest['inputs_unchanged_during_run']
    widths=pd.read_csv(out/'local_width_sites.csv')
    assert widths.groupby('site').size().eq(3).all() and set(widths.threshold_offset)=={-5,0,5}
    assert widths.site.nunique()==31
    assert len(pd.read_csv(out/'hysteresis_sites.csv'))==31
    assert len(manifest['rois'])==11 and all(r['review_status']=='unreviewed' for r in manifest['rois'])
    for m in data['metrics']:
        if m['id'] in ('void_local_width_d50_px','void_local_width_d90_px','crack_local_width_d50_px'):
            update_status(m,'computed','E24: full-site nominal and ±5-level extraction complete; geometric tests pass.',
                          'geometry_tests_passed_expert_pending' if m['review_status']=='unreviewed' else None)
    for method in data['methods']:
        if method['id'] in ('segmentation_benchmark','local_void_width','bright_hysteresis','connectivity_sensitivity'):
            update_status(method,'built' if method['id']=='segmentation_benchmark' else 'computed',
                          'E24 artifacts generated and output contracts verified. Expert review still pending.')
    data['updated']=date.today().isoformat()
    REGISTER.write_text(json.dumps(data,indent=2)+'\n')


def render(data):
    def clean(s):return str(s).replace('|','\\|').replace('\n',' ')
    statuses=Counter(m['implementation_status'] for m in data['metrics'])
    text=[
        '# Morphology metric register',
        '',
        'Generated from `analysis/morphology/metric_register.json`. Edit that JSON to maintain statuses, reviews and history; regenerate this document and the visual atlas. Never edit only the generated HTML/Markdown.',
        '',
        '[Visual metric atlas](../analysis/morphology/output/metric_atlas.html) · [Expert annotation pack](../analysis/morphology/benchmark/review.html) · [E18 morphology results](../analysis/morphology/output/report.html)',
        '[Battery application review](battery_microstructure_review.md): fresh graphite–Si/SiOx material is human-confirmed; mechanism hypotheses remain separate from measurement validation and verdict roles.',
        '[Provider clarification and planned OOD extension](qc_plan.md): Batch 3 is the supplier\'s promised distribution. Missing battery-harm labels is not a universal gate for distribution features. Current metric roles below are unchanged; the bounded extension is planned, not activated (D49P).',
        '',
        '**Implementation, evidence, expert review and QC role are separate fields.** Computed does not mean expert-validated. Only the existing five primary KPIs carry verdicts. All new methods were developed on known batches; none has unseen-batch validation. Sites remain the statistical units, with specimen independence unresolved.',
        '',
        f"Updated {data['updated']}; {len(data['metrics'])} entries (ten-bin profiles and conditional fraction families list every underlying key). Implementation: "+', '.join(f'{k} {v}' for k,v in sorted(statuses.items()))+'.',
        '',
        '| Metric / exact keys | Units | Implementation | Evidence | Expert review | QC role | Experiment | Next check |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for m in data['metrics']:
        next_check=m['review_notes'] or ('Expert measurement review' if m['review_status']=='unreviewed' else 'See review notes and history')
        if m['evidence_status']=='confounded':
            next_check=('E32: no qualifying normalisation; expert crop review and independent acquisition controls'
                        if 'E32' in m['experiments'] else 'Normalisation + acquisition sensitivity; independent review')
        if m['implementation_status'] in ('proposed','deferred','implemented'):next_check='Independent measurement/model validation before numeric use'
        if m['evidence_status'] in ('undefined_on_current_data','constant_on_current_data'):next_check='No per-site discrimination available; do not promote'
        text.append('| '+' | '.join(clean(x) for x in [m['label']+' — '+', '.join('`'+k+'`' for k in m['keys']),m['units'],m['implementation_status'],m['evidence_status'],m['review_status'],m['qc_role'],', '.join(m['experiments']),next_check])+' |')
    text+=['','## Methods and benchmark status','','| Method | Implementation | Evidence | Scope / next step |','|---|---|---|---|']
    for m in data['methods']:
        text.append('| '+' | '.join(clean(x) for x in [m['label'],m['implementation_status'],m['evidence_status'],m['description']])+' |')
    if data.get('battery_checks'):
        text+=['','## Battery hypothesis checks','','These are application hypothesis checks. E25 supplies some secondary geometry measurements; none is an additional primary KPI or expert-validated battery mechanism. Exact chemistry, recipe, collector direction and 3-D/electrical behaviour remain unresolved. The canonical JSON maps each check to existing metric IDs and records its validation needs.','','| Check | Priority | Status | Measurement / review next step |','|---|---|---|---|']
        for c in data['battery_checks']:
            text.append('| '+' | '.join(clean(x) for x in [c['id']+' — '+c['label'],c['priority'],c['status'],c['proposed_action']])+' |')
    text+=['','## Maintenance','','1. Add a new metric with a stable ID, exact keys, definition, units, risks, evidence files, status and history in the JSON register. Do not rename previous IDs.',
           '2. Record expert review independently (`unreviewed`, `confirmed`, `refuted`, `needs_revision`) with notes. Keep older history entries. Algorithm agreement is not validation.',
           '3. Run `python -m analysis.morphology.metric_register` and `python -m analysis.morphology.build_metric_atlas` to regenerate both views.',
           '4. Log a new experiment and append numerical results to the experiment registry. Promoting a KPI requires a separate documented decision and validation; the renderer cannot promote it.',
           '',
           'Expert annotation files are downloaded from the review page and evaluated with `python -m analysis.morphology.build_benchmark --annotations /absolute/path/annotations.json`. Unreviewed crops are not scored; uncertain pixels are ignored; partial labels cannot support component-size or width errors. Development and held-out-site results must remain separate.',
           '']
    (ROOT/'docs/morphology_metrics.md').write_text('\n'.join(text))
    fields=['id','label','family','implementation_status','evidence_status','review_status','qc_role']
    pd.DataFrame([{k:m[k] for k in fields} for m in data['metrics']]).to_csv(REGISTER.parent/'output/metric_status_snapshot.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sync-benchmark',action='store_true');args=parser.parse_args()
    data=json.loads(REGISTER.read_text())
    if args.sync_benchmark:sync_benchmark(data)
    validate(data);render(data)
    print(f"Register validated/rendered: {len(data['metrics'])} metrics; five primary; all E18 descriptors covered.")


if __name__=='__main__':main()
