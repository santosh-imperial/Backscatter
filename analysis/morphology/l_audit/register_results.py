"""Additive E31L documentation/inventory update; run after artifact verification.

This delivery helper never edits production, past registry rows, manual reviews,
feature definitions or prior history items. Concurrent edits cause an explicit stop.
"""
import argparse,csv,hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd

DIRECTORY='analysis/morphology/l_audit'
CAND=['bright_aspect_aw','bright_alignment_strength','bright_pore_crosscorr_xy_contrast_256px']

def sha(data):return hashlib.sha256(data).hexdigest()

def write_checked(p,old,new):
    assert p.read_bytes()==old,('Concurrent update; re-read before additive delivery',p)
    p.write_bytes(new)

DECISION='''### D51L · 2026-10-03 · Complete the morphology OOD audit without automatic promotion
- **Decision:** E31L.1 retains all three candidate measurements/explanations, while automatic OOD input remains deferred by the fixed gates. Threshold/floor, usable-count, fixed acquisition and abstract detection gates pass; incomplete radiometric paired coverage prevents a robustness pass for all three. Directional association additionally lacks supported section orientation. This is an observability limitation, not a demonstrated large movement of valid values. Bright-quality eligibility also affects the existing bright KPIs. No requirement for battery-harm labels is added.
- **Model/error evidence:** the Holm-controlled two-panel union detects fixed candidate-only Gaussian summary shifts better than the primary-only comparator. Its seven-query IID interval narrowly misses the fixed upper precision bound; retain the criterion rather than add trials or move it after results. Naive OR has an extra alert opportunity and is not selected. Known nominal comparisons do not establish separation, but the descriptive Batch 1 phase-association shift remains visible. None establishes acceptance, defect accuracy or held-back performance.
- **Implementation/provenance:** full measurable Batch 3, including all three known long-void sites; shared-site panel comparison and explicit native-coverage sensitivity. All reference fitting is allocation/fold-local; below-minimum deletion folds abstain. Site ranks remain descriptive, not conformal/membership probabilities. Exact internal splits overlap and are calibrated by construction. Both fixed candidate shift signs must pass the added-detection gate. Raw transposes retain the same trimmed ROI. A DataFrame column-lookup repair is recorded with original source/receipts; numeric settings and simulation values are unchanged.
- **Ownership/delivery:** Codex completes **L audit**; Claude retains L1–L4 categorisation, baseline score, Inlens and signatures. Artifacts and register/log updates are additive. No live pipeline, original cache, notebook, report or primary family is changed. Promotion, a new release freeze and final H rehearsal are separate actions. C01–C07/C09–C11/C13–C17/C19–C21/C23/C24/C26/C28–C30.
'''

EXPERIMENT='''### E31L · 2026-10-03 · Registered morphology OOD promotion audit (L audit)
- **Scope/provenance:** E31L.1 registered in D50L before new outputs; known Batches 1–3 were already explored. Fresh replay on 31 BSE sites at paired −5/0/+5 thresholds; 124 site-modes including nominal floor100. K’s hash-bound raster association is reused and independently replayed at four fixed examples × three thresholds. Four same-ROI raw transposes and 12 radiometric copies are saved. No held-back/unseen data, production input or verdict changes.
- **Coverage:** aspect/alignment use 5/7/17 sites in B1/B2/B3; shared cross-phase comparison uses 5/7/13, retaining all three known long-void reference sites. Components/pixel pairs are coverage, not n. Below-five-site deletion folds have unavailable alert status. Specimen independence is unknown.
- **Known comparison:** shared nominal primary5 energy p=0.313725/0.0732, morphology3 p=0.206116/0.7592, extended8 p=0.256303/0.1522 for B1/B2. Equal panel weights differ from the frozen consequence weights. B1 phase-association raw median shift +0.0189433 (4,000 approximate site-bootstrap interval +0.00547772 to +0.0323513), eight-feature Holm-adjusted HL p=0.098973. No significant omnibus departure or equivalence/acceptance is claimed; signatures remain descriptive.
- **Candidate gates:** all threshold/floor modes pass fixed median≤0.5/max≤2 reference-MAD movement gates. Acquisition LOO R² = −0.463848 / +0.003826 / −0.242237 for aspect/alignment/association; poor prediction is not invariance. Valid radiometric max movement = 0.029368 / 0.007445 / 0.373225 MAD, but only 6/9, 6/9 and 4/6 source-eligible cases remain available after transformed quality flags. Automatic promotion is deferred for incomplete observability; association’s section-direction gate is also unresolved. Existing bright KPIs share the quality-eligibility issue.
- **Joint error/power controls:** 200 independent correlated Gaussian draws at each 13/5 and 13/7 size, 499 Monte Carlo allocations each, re-fit reference per allocation. Naive OR alert fraction 0.10/0.11; Holm union 0.04/0.06, Wilson intervals [0.020406,0.076932]/[0.034652,0.101932]. The second upper bound narrowly exceeds the fixed≤0.10 gate; no extra trials or threshold relaxation. All 3,003 disjoint within-reference allocations are enumerated; overlapping exact ranks are internally calibrated, not independent/external error estimates. Eight fixed feature-space controls use 120 independent paired draws each (960 total); union gains over primary5 for single-candidate ±2 population-scaled-MAD shifts are 0.575–0.666667 with positive paired intervals. These are abstract summary changes, not physical defects or manufacturing accuracy.
- **Interpretability/limits:** 3NN scores, reference-LOO descriptive ranks, matched-count query sensitivity and nearest-reference feature contributions are saved. Whole-site acquisition/geometry ridge controls use train-fold imputation/scaling; within-batch correlations are descriptive. No calibrated site probability, classifier posterior, chemical-instance validation, transport or battery-performance claim. Claude’s categoriser/Inlens/signature work is separate.
- **Execution repair/verification:** the first run stopped in robustness assembly because the `transform` column collided with a pandas method; bracket lookup only was fixed, original source/receipts preserved, completed raw/simulation outputs reused unchanged. Twenty-two L/K convention tests pass (11 new L tests); independent verifier checks source/raw/table hashes, all nominal replays, coverage/long-void retention, pair counts, axes, known tests, abstentions, 400 IID and 960 power trials, exact allocations, paired gains, scores/contributions and report links. One harmless future dtype warning is confined to a test fixture.
- **Delivery:** `analysis/morphology/l_audit/report.html`, `recommendations.csv`, `handoff.md`, frozen protocol, measured/control tables, source snapshots, repair lineage, verification and delivery receipts. Three existing metric histories receive evidence links; one computed audit method is added. All live feature roles and manual review states are preserved. D51L records the recommendation. C01–C07/C09–C11/C13–C17/C19–C21/C23/C24/C26/C28–C30.
'''

FINDINGS='''## Morphology OOD audit completed (L audit / E31L / D51L)

The standalone [E31L report](../analysis/morphology/l_audit/report.html) evaluates the bounded D49P morphology panel while Claude handles the L1–L4 sample deliverables. All 31 nominal primary/shape replays agree with saved measurements. Threshold ±5 and floor100 gates pass for aspect, axial strength and the 256 px phase-association contrast; fixed acquisition/redundancy screens do not fail their gates, but weak predictability is not material-origin proof.

Automatic OOD input is deferred by specific unresolved gates, not missing battery-harm labels. Valid brightness-control pairs move little (maximum 0.0294/0.00745/0.3733 reference MAD), but transformed quality flags remove coverage: 6/9 aspect/alignment and 4/6 association source-eligible pairs remain measurable. This quality-policy sensitivity also affects existing bright KPIs. Directional association additionally needs section/collector orientation. Retain these measured descriptors/explanations; register any revised quality policy or shadow model separately.

Shared nominal primary5 and extended8 omnibus tests do not establish departure for B1/B2; B1 association has a +0.01894 median shift (approximate site interval +0.00548 to +0.03235; eight-feature-adjusted HL p≈0.099). This is descriptive batch-signature evidence, not proof of sameness or a forced OOD label. The common usable reference has 13 B3 sites including all three long-void sites, with 5/7 B1/B2; quality losses and deletion abstention remain visible.

The combined rule matters: naive OR fires on 10%/11% of IID null trials, Holm union on 4%/6%. The seven-query union interval’s upper limit 0.10193 narrowly misses the prespecified≤0.10 gate; it does not demonstrate the true rate exceeds 10%. Abstract candidate-only controls show positive added detection, not manufacturing accuracy. All 3,003 overlapping internal reference allocations are diagnostic ranks, not independent external false-alert trials. Site distances/LOO percentiles are descriptive and uncalibrated. Twenty-two L/K convention tests and artifact checks pass; no live verdict input or genuine unseen validation was added. [Handoff](../analysis/morphology/l_audit/handoff.md).
'''

def register(root):
    out=root/DIRECTORY
    assert (out/'verification.json').exists(),'Verify artifacts before delivery bookkeeping'
    register_path=root/'analysis/morphology/metric_register.json';old=register_path.read_bytes();data=json.loads(old)
    assert not any(x['id']=='morphology_ood_audit' for x in data['methods']),'Already registered; do not duplicate'
    previous=json.loads(old)
    for entry in data['metrics']:
        if entry['id'] in CAND:
            for p in [DIRECTORY+'/protocol.json',DIRECTORY+'/report.html',DIRECTORY+'/recommendations.csv']:
                if p not in entry['evidence_paths']:entry['evidence_paths'].append(p)
            if 'E31L' not in entry['experiments']:entry['experiments'].append('E31L')
            entry['history'].append(dict(date='2026-10-03',status=entry['implementation_status'],
                reason='E31L: registered OOD audit complete; automatic input deferred by observability/section gates. Definitions, manual review and QC role unchanged.'))
    data['methods'].append(dict(id='morphology_ood_audit',label='Registered morphology OOD promotion audit',
        implementation_status='computed',evidence_status='known_site_and_engineering_controls_automatic_use_deferred',experiment='E31L',
        description='Equal-weight primary/morphology energy panels, allocation-local reference fits, joint Holm alert control, quality/movement/axis gates and descriptive 3NN explanations. Gaussian controls are not defect truth; no live promotion.',
        history=[dict(date='2026-10-03',status='computed',reason='E31L.1 completed; threshold/floor and abstract added-detection pass; radiometric observability/axis and joint precision gates remain. No production/classifier/verdict input change.')]))
    data['updated']='2026-10-03'
    for a,b in zip(previous['metrics'],data['metrics']):
        assert a['definition']==b['definition'] and a['qc_role']==b['qc_role'] and a['review_status']==b['review_status']
        assert a['history']==b['history'][:len(a['history'])]
    assert previous['methods']==data['methods'][:len(previous['methods'])]
    write_checked(register_path,old,(json.dumps(data,indent=2)+'\n').encode())

    fields=['experiment','date','metric','reference','batch','kpi','statistic','n_ref','n_batch','value','unit','note']
    tables=['raw_summaries','coverage','batch_tests','batch_policies','univariate','deletions','reference_diagnostics',
            'iid_summary','power_summary','paired_gains','movement_pairs','movement_summary','radiometric','radiometric_movement',
            'axis_raw','axis_tests','control_scores','control_predictions','within_batch_correlations','site_scores','site_contributions','pair_counts','phase_replay','recommendations']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');nrows=0
    for name in tables:
        t=pd.read_csv(out/(name+'.csv'))
        numeric=[c for c in t if pd.api.types.is_numeric_dtype(t[c])]
        for _,row in t.iterrows():
            context={c:row[c] for c in ['site','mode','view','panel','policy','scenario','transform','family','covariate','kind','axis','lag_px','deleted_role','deleted_site','query_n'] if c in row and pd.notna(row[c])}
            for c in numeric:
                if pd.isna(row[c]):continue
                unit='indicator' if pd.api.types.is_bool_dtype(t[c]) else ('p' if c.startswith('p_') or c in ['p','holm_primary','holm_morphology','p_holm8','query_only_swap_p','nominal_p','global_axis_swap_p'] else 'result_field')
                writer.writerow(dict(experiment='E31L',date='2026-10-03',metric=name,reference='Batch_3' if name not in ['iid_summary','power_summary','paired_gains'] else 'independent_Gaussian_summary_controls',
                    batch=row.get('batch',row.get('scenario','known_sites_or_controls')),kpi=row.get('kpi',c),statistic=c,
                    n_ref=row.get('n_ref',''),n_batch=row.get('n_batch',''),value=float(row[c]),unit=unit,
                    note=json.dumps(context,sort_keys=True)+'; sites are n; pixels/components are coverage; L development audit, no live verdict activation; per-trial null draws saved separately'))
                nrows+=1
    counts={'sites_replayed':31,'site_modes':124,'phase_raw_replays':12,'axis_raw_rows':12,'radiometric_copies':12,
            'reference_allocations':3003,'iid_independent_trials':400,'feature_space_trials':960,
            'new_L_tests_passed':11,'L_K_tests_passed':22,'inventory_metrics':len(data['metrics']),'inventory_methods':len(data['methods']),
            'numerical_registry_fields':nrows}
    for k,v in counts.items():writer.writerow(dict(experiment='E31L',date='2026-10-03',metric='delivery_count',reference='known-data',batch='L audit',kpi=k,statistic='count',n_ref='',n_batch='',value=v,unit='count',note='Administrative/coverage/trial count; not material-independent n or defect accuracy'))
    registry=root/'experiments/registry.csv';old=registry.read_bytes()
    assert b'E31L,' not in old,'Already appended; do not duplicate'
    assert old.endswith(b'\n')
    registry_prefix_hash=sha(old);registry_prefix_bytes=len(old)
    write_checked(registry,old,old+stream.getvalue().encode())
    assert registry.read_bytes().startswith(old)

    decision=root/'docs/decision_log.md';old=decision.read_bytes();text=old.decode()
    if '### D50L ' not in text:
        text=text.replace('## Part B — Pre-presentation review checklist',
            '### D50L · 2026-10-03 · Register morphology OOD audit E31L.1 before new results\n'
            '- **Decision:** the timestamp/hash-registered protocol in `analysis/morphology/l_audit/` fixes the bounded three-candidate panel, full measurable reference, allocation-local fitting, joint-alert/coverage/movement/axis/error/detection gates and descriptive site explanations before new L results. Known data were already inspected. Claude owns categorisation/Inlens; no live verdict activation. Original registration made in isolated `codex/morphology-l-audit`; copied here additively after execution. C01–C07/C09/C10/C13/C14/C16/C17/C21/C23/C26/C28–C31.\n\n'
            '## Part B — Pre-presentation review checklist')
    assert '### D51L ' not in text
    text=text.replace('## Part B — Pre-presentation review checklist',DECISION+'\n## Part B — Pre-presentation review checklist')
    write_checked(decision,old,text.encode())
    log=root/'docs/experiment_log.md';old=log.read_bytes();text=old.decode();assert '### E31L ' not in text
    text=text.replace('## Part C — Next experiments (queued)',EXPERIMENT+'\n## Part C — Next experiments (queued)')
    line='| | morphology OOD promotion audit | ◑ | E31L: 31 fresh replays; 22 L/K tests; joint alerts/detection/coverage/site explanations saved | automatic input deferred by radiometric observability, section direction and joint-rate precision gates; unseen untested |\n'
    anchor='| | shape variability / directional phase arrangement |'
    lines=text.splitlines(keepends=True)
    for i,s in enumerate(lines):
        if s.startswith(anchor):lines.insert(i+1,line);break
    else:raise AssertionError('Missing scoreboard anchor')
    write_checked(log,old,''.join(lines).encode())
    findings=root/'docs/problem_and_findings.md';old=findings.read_bytes();text=old.decode()
    assert '## Morphology OOD audit completed' not in text
    text=text.replace('the morphology OOD extension in `qc_plan.md` §1.1 is planned, not implemented or validated.',
                      'the standalone morphology OOD development audit in `qc_plan.md` §1.1 is completed as E31L, while live promotion is deferred and genuine unseen validation is unavailable.')
    write_checked(findings,old,(text.rstrip()+'\n\n'+FINDINGS).encode())
    plan=root/'docs/qc_plan.md';old=plan.read_bytes();text=old.decode()
    text=text.replace('§1.1 is a planned morphology OOD extension following the provider\'s latest clarification, not an executed model.',
                      '§1.1 has a completed standalone morphology OOD audit (E31L) following the provider\'s clarification; its live feature promotion remains deferred.')
    text=text.replace('### 1.1 Supplier baseline and morphology OOD extension (D49P, planned)',
                      '### 1.1 Supplier baseline and morphology OOD extension (D49P; E31L audit complete)')
    status='\n**E31L status (D51L).** The registered standalone [L audit](../analysis/morphology/l_audit/report.html) is complete, distinct from Claude’s L1–L4 sample deliverables. Candidate movement/acquisition/abstract-detection gates pass; incomplete transformed quality coverage blocks automatic input, and association lacks section orientation. The combined-rule precision gate narrowly remains unmet. Valid feature values are stable in measured brightness controls; missing pairs are not stability evidence. See [the handoff](../analysis/morphology/l_audit/handoff.md). No release/classifier/verdict input changes, tolerance claim or genuine unseen validation was added.\n\n'
    assert '**E31L status' not in text
    text=text.replace('## 2. Notebook sections',status+'## 2. Notebook sections')
    write_checked(plan,old,text.encode())
    tasks=root/'docs/next_steps.md';old=tasks.read_bytes();text=old.decode()
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| **L**') and 'Register and run a bounded morphology OOD' in line:
            lines[i]='| **L audit** | P0 / medium | Registered morphology OOD promotion audit: baseline versus bright aspect/alignment/256 px phase association; full measurable B3, shared coverage, perturbation/acquisition/axis controls, joint alerts and added detection. | Codex engineering; Santosh image interpretation | **Completed E31L.1 (D50L/D51L)** in isolated `codex/morphology-l-audit`; copied additively to main. [Report](../analysis/morphology/l_audit/report.html) · [Handoff](../analysis/morphology/l_audit/handoff.md). Threshold/floor and abstract detection pass; automatic input deferred by incomplete radiometric observability, section direction and joint-rate precision. Claude keeps L1–L4 ownership below; no live activation or commit. |'
    text='\n'.join(lines)+'\n'
    text=text.replace('it is planned, not executed.','the standalone E31L audit is complete, with live promotion deferred; Claude’s sample deliverables remain separate.')
    write_checked(tasks,old,text.encode())
    dump=dict(experiment='E31L',registry_prefix_sha256=registry_prefix_hash,registry_prefix_bytes=registry_prefix_bytes,
              appended_numeric_fields=nrows,counts=counts,prior_metric_histories_preserved=True,prior_methods_preserved=True,
              manual_reviews_and_roles_preserved=True)
    (out/'bookkeeping.json').write_text(json.dumps(dump,indent=2)+'\n')
    print('Additive docs/registry/metric evidence updated:',counts)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);args=parser.parse_args()
    register(args.root.resolve())
