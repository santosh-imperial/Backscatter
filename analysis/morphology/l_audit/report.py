"""Render E31L's saved audit, with no new model or measurement choices."""
import base64,html,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/polaron-l-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .methods import BASE,CAND,KEYS

OUT=Path(__file__).resolve().parent
LABELS={CAND[0]:'Bright aspect ratio (area weighted)',CAND[1]:'Bright axial alignment strength',
        CAND[2]:'Bright/void x−y association at 256 px',
        'crack_frac':'Large elongated void area fraction','pore_max_d':'Maximum void diameter (px)',
        'pore_frac':'Void area fraction','bright_frac':'Bright area fraction','bright_d50':'Bright D50 (px)'}
COLORS={'Batch_1':'#2a78d6','Batch_2':'#eb6834','Batch_3':'#1baf7a'}

def table(t,cols=None):
    if cols is not None:t=t[cols]
    return t.to_html(index=False,escape=True,na_rep='Unavailable',float_format=lambda v:f'{v:.4g}',border=0,classes='data')

def img(path,alt):
    data=base64.b64encode(Path(path).read_bytes()).decode()
    return '<img alt="'+html.escape(alt)+'" src="data:image/png;base64,'+data+'">'

def savefig(fig,name):
    fig.tight_layout();p=OUT/'assets'/name;fig.savefig(p,dpi=140);plt.close(fig);return p

def build():
    def read(name):return pd.read_csv(OUT/(name+'.csv'))
    sites=read('sites');rec=read('recommendations');bt=read('batch_tests');policy=read('batch_policies')
    iid=read('iid_summary');power=read('power_summary');gain=read('paired_gains');mov=read('movement_summary')
    radio=read('radiometric_movement');score=read('site_scores');controls=read('control_scores')
    deletion=read('deletions');uni=read('univariate');internal=read('reference_diagnostics')
    manifest=json.loads((OUT/'manifest.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k in zip(axes,CAND):
        for i,b in enumerate(['Batch_1','Batch_2','Batch_3']):
            g=sites[sites.batch.eq(b)];valid=~g.bright_low_contrast
            if k==CAND[2]:valid &= ~g.grey_pore
            values=g.loc[valid,k].to_numpy()
            x=i+np.linspace(-.12,.12,len(values))
            ax.scatter(x,values,color=COLORS[b],label=b.replace('_',' '),s=24)
            if len(values):ax.plot([i-.18,i+.18],[np.median(values)]*2,color=COLORS[b],lw=3)
        ax.set_xticks(range(3),['Batch 1','Batch 2','Batch 3']);ax.set_title(LABELS[k],fontsize=10)
        ax.set_ylabel('Ratio' if k==CAND[0] else ('Strength (0–1)' if k==CAND[1] else 'Correlation contrast'))
        ax.grid(axis='y',alpha=.2)
    strips=savefig(fig,'candidate_sites.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,nb in zip(axes,[5,7]):
        g=iid[iid.query_n.eq(nb)];x=np.arange(len(g))
        ax.bar(x,g.rate,color=['#4487bb','#b88639','#168668','#ba4a54'])
        ax.errorbar(x,g.rate,yerr=[g.rate-g.ci_low,g.ci_high-g.rate],fmt='none',ecolor='black',capsize=4)
        ax.axhline(.05,color='#555',ls='--',label='Nominal α=.05');ax.set_ylim(0,.18)
        ax.set_xticks(x,g.policy,rotation=18);ax.set_title(f'IID summary-space null: 13 reference / {nb} query')
        ax.set_ylabel('Alert fraction across 200 independent draws');ax.legend(fontsize=8)
    nullplot=savefig(fig,'iid_null.png')
    fig,ax=plt.subplots(figsize=(12,5))
    scenarios=power.scenario.unique();x=np.arange(len(scenarios));width=.23
    for j,p in enumerate(['primary5','extended8','holm_union']):
        g=power[power.policy.eq(p)].set_index('scenario').loc[scenarios]
        ax.bar(x+(j-1)*width,g.rate,width=width,label=p)
        ax.errorbar(x+(j-1)*width,g.rate,yerr=[g.rate-g.ci_low,g.ci_high-g.rate],fmt='none',ecolor='#333',capsize=2,lw=.8)
    ax.set_ylim(0,1);ax.set_ylabel('Detection fraction across 120 independent draws')
    ticklabels=[]
    for s in scenarios:
        if '__' in s:
            key,sign=s.split('__');short={CAND[0]:'aspect',CAND[1]:'alignment',CAND[2]:'association'}[key]
            ticklabels.append(short+': '+('−2' if sign=='-1' else '+2')+' scaled MAD')
        else:ticklabels.append('Candidate spread ×2' if s=='candidate_spread_x2' else 'Primary5: +2 scaled MAD')
    ax.set_xticks(x,ticklabels,rotation=25,ha='right')
    ax.set_title('Abstract feature-space controls (not manufacturing defects)');ax.legend()
    powerplot=savefig(fig,'controlled_detection.png')
    sections=[]
    def add(title,body):sections.append('<section><h2>'+html.escape(title)+'</h2>'+body+'</section>')
    def p(s):return '<p>'+s+'</p>'
    n_available=int((~sites.bright_low_contrast&~sites.grey_pore).sum())
    add('Decision for this development version',p('This audit evaluates the bounded morphology expansion proposed in D49P. It does not run Claude’s three-class categoriser, calibrate his site-level baseline score, or change the frozen QC verdict. All known sites were previously explored; no held-back or unseen images were read.')+
        table(rec,['kpi','recommendation','unresolved_gates','joint_model_gate_pass'])+
        p('All three candidates pass the fixed threshold/floor, coverage, acquisition-predictability and abstract added-detection gates. Valid radiometric pairs move less than the registered limit, but changed quality flags leave incomplete paired coverage: 6/9 cases for aspect/alignment and 4/6 for association. The unresolved radiometric gate concerns observability under those flags, not a demonstrated large movement of the valid candidate values. Bright-feature eligibility also affects existing bright KPIs; it is not unique to these additions.')+
        p('A failed gate is a concrete reason to defer automatic OOD input. A passing candidate would move only to versioned shadow development. Independent mask review, specimen/session repeatability and a separate unseen evaluation remain required evidence. Battery-harm labels are not a gate for detecting a distribution change.'))
    add('Reference and coverage',p(f'The supplier-promised Batch 3 reference keeps its measurable heterogeneity, including all three known long-void sites. All {len(sites)} known sites were replayed; the shared eight-feature view has {n_available} usable sites (Batch 1: 5, Batch 2: 7, Batch 3: 13). Low bright contrast removes two Batch 1 sites; grey-pore observability removes four Batch 3 sites from cross-phase comparison. Each excluded site remains in the saved table. Component/pixel counts are measurement coverage, never statistical n.')+
        table(read('coverage').query("mode == 'nominal' and view == 'native'"),['panel','batch','n_folder','n_usable','n_abstained'])+
        p('The primary-only native view retains grey-pore measurements as a disclosed sensitivity view, reflecting the existing soft pore flag. The main panel comparison uses the common usable rows. Reference MAD is fitted separately for every allocation; zero-MAD columns use reference SD, then unit scale. Fallback counts accompany every test. All lengths are pixels; scale metadata remain unverified.'))
    add('Known-batch comparison',p('Energy comparisons use equal feature weights within each panel. This is a development comparator, distinct from the frozen consequence-weighted QC rule. Nonfinite inputs and unknown quality abstain. A nonsignificant result does not establish equivalence or acceptance. Threshold/floor modes are paired sensitivity views, not independent replication.')+
        table(bt.query("mode == 'nominal' and view == 'shared'"),['batch','panel','n_ref','n_batch','statistic','p','method','n_allocations','fallback_sd','fallback_unit'])+
        table(policy.query("mode == 'nominal'"),['batch','p_primary5','p_morphology3','p_extended8','holm_primary','holm_morphology','primary5','extended8','holm_union','naive_or'])+
        p('Primary5 alone and extended8 alone each have one omnibus opportunity. The combined rule is the Holm-controlled union of primary5 and morphology3. The naive OR is displayed to expose the extra alert opportunity; it is not the recommendation. The eight explanatory location tests are not additional alert triggers.'))
    add('What differs in the measured geometry',img(strips,'Each candidate at every usable known site, with batch medians')+
        p('Gold/blue masks are algorithm predictions. Bright fragments are observed BSE appearance, not chemically confirmed individual Si/SiOx particles. Shape and alignment describe section geometry; directional association is a binary finite-domain correlation difference, not electrical contact or 3-D transport.')+
        table(uni,['batch','kpi','n_ref','n_batch','median_difference','ci_low','ci_high','hl_difference','p','p_holm8'])+
        p('Intervals are 4,000 percentile bootstraps over sites for the raw median difference, approximate with these small counts. Holm correction covers the eight Hodges–Lehmann location tests within each batch contrast. They remain development evidence because the known batches informed earlier work. Shared specimens could make site uncertainty too optimistic.'))
    add('Images and component eligibility',img(OUT/'assets/examples.png','Four prespecified real BSE examples, phase masks and elongated bright-component axes')+
        p('Left: trimmed source. Middle: bright gold and void blue masks. Right: major-axis direction indicators for nonclipped bright components with area ≥50 px² and aspect ≥1.5, placed at bounding-box centres; they are visual direction guides, not traced particle boundaries. Alignment needs at least 20 oriented components; aspect needs 20 interior components. Ordinary, low-contrast, grey-pore and long-void cases were fixed before the run. Low-contrast/grey-pore examples illustrate abstention and do not validate masks.'))
    add('Threshold, object-floor and radiometric robustness',table(mov,['kpi','mode','n_source','n_pairs','median_movement_mad','max_movement_mad','gate_pass'])+
        p('The fixed engineering gate is median absolute movement ≤0.5 reference MAD and maximum ≤2 MAD in every mode, with full paired coverage. Floor100 changes only shape/alignment. Its raster association is an exact duplicate, not an extra robustness experiment.')+
        table(radio,['batch','site','kpi','transform','source_available','transformed_available','paired_available','abs_delta_mad'])+
        p('Gamma 0.75/1.25 and rounded 0.8I+20 were applied to copies of the four prespecified raw BSE examples, then trimming/adaptive segmentation were rerun. The latter has no intensity clipping but can change quality flags. Unavailable pairs do not pass a robustness gate. Numerical movement on valid pairs is reported; invariance would not prove a material origin, and loss of separation would not prove an acquisition origin. Full transformed thresholds, dimensions and flags are saved.'))
    add('Image direction and section metadata',table(read('axis_tests'))+
        table(read('axis_raw'),['batch','site','kpi','nominal','transposed','expected','absolute_error'])+
        p('Fresh transposes use the same trimmed ROI and thresholds, preserving the physical pixels. Shape and axial strength are axis-invariant; x−y association changes sign. A global sign change leaves distances/tests unchanged. Query-only axis interchange represents unsupported section orientation and can alter comparison evidence. The association candidate therefore has an unresolved axis gate until section/collector orientation is established. This is not a material-change control.'))
    add('Joint false-alert checks',img(nullplot,'IID false-alert rates with Wilson intervals for independent Gaussian draws')+
        table(iid)+p('The 200 draws at each sample size are independent correlated Gaussian summary-space samples, with reference fitting and 499 Monte Carlo allocations per trial. Wilson intervals apply to these draws only. The model gate requires the Holm-union interval to include 0.05 and its upper limit to be ≤0.10 at both query sizes. This checks the defined comparator, not all rules of the production QC pipeline or real supplier-session error.')+
        p('Holm-union rates are 0.04 (interval 0.0204–0.0769) at five query sites and 0.06 (0.0347–0.1019) at seven. The latter narrowly misses the prespecified precision gate; it does not demonstrate that the true rate exceeds 0.10. Do not raise the bound or add trials to this version after seeing the result. Any larger precision run must be registered separately.')+
        table(internal,['query_n','n_ref','policy','n_allocations','n_alerts','alert_fraction','n_abstained'])+
        p('The reference diagnostic enumerates every disjoint allocation within all 13 shared usable Batch 3 sites, including long voids. Exact allocation ranks have internal calibration by construction. Splits overlap and are not independent false-alert trials, so no binomial confidence interval is attached. They do not establish the reference is homogeneous.'))
    add('Added detection in controlled summary space',img(powerplot,'Detection fractions under fixed abstract changes to summary coordinates')+
        table(gain[gain.policy.isin(['holm_union','extended8'])],['scenario','policy','n_trials','mean_gain','ci_low','ci_high'])+
        p('Each single candidate is shifted ±2 generating-population scaled MAD; the Gaussian scale is 1. Candidate spread is doubled in its three-coordinate block; the final control shifts all five primary coordinates +2. These summaries are abstract engineering controls, not physically valid battery defects. Gains pair the policies on the same independent trial, with 4,000 trial-bootstrap intervals. Both fixed shift directions must meet a candidate’s +0.10 gain/lower-bound&gt;0 gate; no best direction is selected. Single-candidate ablations and their intervals are saved alongside the union.'))
    add('Acquisition and redundancy screens',table(controls)+p('Whole-site leave-one-out Ridge (α=1) uses train-fold-only median imputation and scaling. Acquisition has six fixed covariates; geometry/loading has the five primary KPIs plus bright D90/circularity/density. Targets use their own eligible sites; within-batch correlations and all fold predictions are saved. Acquisition R²&gt;0.25 fails its fixed gate; geometry/loading R²&gt;0.75 warns of redundancy. Negative R² or weak correlations do not establish independence, invariance or added value. These regressions provide associations, not attribution of a microscope share.'))
    add('Site-level explanations',table(score[score.panel.eq('extended8')],['batch','site','n_fit','score_3nn','reference_loo_rank','matched_count_score_median','matched_count_score_min','matched_count_score_max','nearest_reference'])+
        p('Score = mean distance to the three nearest reference sites after reference median/MAD fitting and equal panel weights. Reference rows are whole-site leave-one-out; incoming rows use full reference. The matched-count query view repeats all reference deletions. Rank is a descriptive percentile against reference LOO scores, not a calibrated membership probability, conformal p-value, classifier posterior or forced in/out label. Scores based on different reference counts have unequal fitting uncertainty. Each feature’s squared contribution to the nearest reference distance is saved; it explains that distance, not why a battery will fail.'))
    folds=deletion.groupby(['batch','panel'],sort=False).agg(n_folds=('deleted_site','size'),n_available=('alert_available','sum'),
             p_min=('p','min'),p_max=('p','max'))
    add('Small-dataset sensitivity and next action',table(folds.reset_index())+
        p('Each incoming/reference deletion repeats reference fitting and the test. Removing a usable Batch 1 site leaves only four incoming sites: its alert status is unavailable, even when a numerical p-value is computed for sensitivity. Do not count those folds as stable negatives. All deletion p-values, available alerts and union statuses are saved.')+
        p('For Claude’s integration: preserve separate classifier, baseline-distance and release-decision outputs. Treat this audit as evidence for a future versioned OOD feature decision. Resolve failed movement/observability and axis gates with repeat acquisition, independent masks and section metadata; then register any revised measurement once. Do not sweep thresholds or retune to separate Batches 1/2. Freeze before the genuine unseen run and save that result before further exploration.'))
    files=['protocol.json','protocol_registration.json','manifest.json','recommendations.csv','sites.csv','variants.csv','batch_tests.csv',
           'batch_policies.csv','univariate.csv','deletions.csv','reference_diagnostics.csv','reference_allocations.csv','iid_summary.csv',
           'iid_trials.csv','power_summary.csv','paired_gains.csv','power_trials.csv','site_scores.csv','site_contributions.csv',
           'movement_summary.csv','movement_pairs.csv','radiometric.csv','radiometric_movement.csv','axis_raw.csv','axis_tests.csv',
           'control_scores.csv','control_predictions.csv','within_batch_correlations.csv','pair_counts.csv','phase_replay.csv','README.md','handoff.md']
    add('Reproducibility and saved evidence',p('E31L.1 • base 87ecee3 • registered '+html.escape(manifest['registered_utc'])+
        ' • protocol SHA256 '+html.escape(json.loads((OUT/'protocol_registration.json').read_text())['protocol_sha256'])+
        '. Raw/source/table hashes and snapshots bind warm K association reuse to the fresh raw replay. Original caches and production files are unchanged. Rerun commands and verification are in the README.')+
        p('Execution repair: the first run saved raw measurements, batch tests and simulations, then stopped because the DataFrame column “transform” collided with a pandas method. The repair uses bracket lookup only; original source and receipts are preserved alongside the repaired lineage. Measurement/test settings and saved simulation numbers did not change.')+
        '<ul>'+''.join('<li><a href="'+f+'">'+f+'</a></li>' for f in files)+'</ul>'+
        p('Review applied: C01–C07/C09–C11/C13–C17/C19–C21/C23/C24/C26/C28–C30. No defect accuracy, manufacturing tolerance, calibrated site probability, physical contact, performance percentage or unseen-generalisation claim.'))
    css='body{font:16px/1.5 system-ui,sans-serif;color:#23313d;background:#eef3f5;margin:0}main{max-width:1320px;margin:auto;padding:26px}section{background:white;padding:24px;margin:20px 0;border-radius:12px;overflow:auto}h1{font-size:30px}h2{font-size:22px;color:#173d52}p{max-width:1080px}.data{border-collapse:collapse;font-size:13px;width:100%}td,th{text-align:left;padding:7px;border-bottom:1px solid #dde5e9}th{background:#e9f0f2}img{width:100%;height:auto}a{color:#13678a}.badge{display:inline-block;background:#d7e9ef;padding:5px 12px;border-radius:18px}'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E31L morphology OOD audit</title><style>'+css+'</style><main><span class="badge">Development audit • no live activation</span><h1>Does the morphology expansion earn an OOD role?</h1><p>Registered measurement, robustness, joint-alert and added-detection audit on the known supplier batches.</p>'+''.join(sections)+'</main></html>'
    (OUT/'report.html').write_text(page)
    print('Illustrated report saved:',OUT/'report.html',flush=True)

if __name__=='__main__':build()
