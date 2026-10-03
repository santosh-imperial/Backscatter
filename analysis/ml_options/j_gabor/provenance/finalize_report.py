"""Qualify fixed measurements without rerunning extraction or changing settings."""
from pathlib import Path
import hashlib
import html
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    comp=pd.read_csv(OUT/'comparisons.csv')
    sites=pd.read_csv(OUT/'sites.csv')
    pred=pd.read_csv(OUT/'control_predictions.csv')
    sens=pd.read_csv(OUT/'sensitivity_summary.csv')
    nominal=comp[comp['mode'].eq('nominal')]
    matched=nominal[nominal.view.eq('quality_matched')]
    assert len(matched)==9 and ((matched.ci_low<=0)&(matched.ci_high>=0)).all()
    def contrast(view):
        return nominal[nominal.view.eq(view)&nominal.kpi.eq('gabor_coarse_energy_share')&nominal.batch.eq('Batch_1')&nominal.reference.eq('Batch_3')].iloc[0]
    a,b=contrast('all_known'),contrast('quality_matched')
    r2=pred.groupby(['view','kpi','control']).loo_r2.first()
    assert (r2<0).all()
    res=sens[sens.view.eq('all_known')&sens['mode'].eq('full_resolution')]
    affine=sens[sens['mode'].eq('affine')]
    assert affine.max_abs_delta.max()<1e-12
    reading=f'''# Measured qualification · task J / E28J

Retain the response maps for exploratory BSE appearance review. Defer all
three summaries as material/QC drivers. No evidence here establishes useful
information beyond existing geometry or acquisition variables.

The all-known Batch1−Batch3 coarse-energy-share difference is
{a.median_difference:+.5f}, site-bootstrap95% [{a.ci_low:+.5f}, {a.ci_high:+.5f}]
at n{int(a.n_batch)}/{int(a.n_reference)} sites. In the quality-matched view it is
{b.median_difference:+.5f} [{b.ci_low:+.5f}, {b.ci_high:+.5f}], n{int(b.n_batch)}/{int(b.n_reference)}.
All nine nominal quality-matched intervals include zero. Filtering changes
both the population and sample size; this does not prove the difference was
an acquisition artifact or that the batches are equivalent. These are
correlated exploratory contrasts, with no multiplicity-adjusted test.

Nominal versus full-resolution ranks have Spearman rho
{res.spearman_nominal_mode.min():.3f}–{res.spearman_nominal_mode.max():.3f}; median
absolute descriptor movement is {res.median_delta_over_site_iqr.min():.3f}–{res.median_delta_over_site_iqr.max():.3f}
of the pooled between-site IQR. Gamma changes move some sites more; complete
paired values and maxima are in `sensitivity_summary.csv`. The non-clipping
affine control agrees to numerical precision, as expected by construction.
This is not evidence that real contrast loss or preparation is corrected.

All twelve acquisition and FFT/geometry/tensor site-LOO ridge R² results are
negative. That means these prespecified linear controls predict poorly at
this n; it does not establish new information or acquisition invariance.
Strong within-batch correlations remain in the correlation table, with
small group counts and different sampled versus full-frame measurands.

Valid convolution interiors cover only {100*sites.gabor_valid_area_fraction.min():.2f}–{100*sites.gabor_valid_area_fraction.max():.2f}%
of each trimmed frame. Four windows are coverage, not independent samples;
spatial representativeness and specimen independence remain unvalidated.
Wavevectors are normal to image stripes, not graphite plate-instance axes.
Expert phase validity, battery mechanisms, manufacturing outcomes and
unseen generalisation remain open. Primary/classifier/verdict inputs are
unchanged.

Eight mathematical convention/invariance tests pass. They do not validate
material identity or independent segmentation/defect accuracy.
Review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.
'''
    (OUT/'review.md').write_text(reading)
    findings=OUT/'findings.md'
    s=findings.read_text().replace('The full-frame sampling mask and valid-area fractions are saved per site.',
                                 'The full-frame window coordinates and valid-area fractions are saved per site.')
    s=s.split('\n## Post-run measured qualification')[0].rstrip()
    findings.write_text(s+'\n\n## Post-run measured qualification\n\n'+reading+'\n')
    report=OUT/'report.html'
    s=report.read_text().replace('The full-frame sampling mask and valid-area fractions are saved per site.',
                               'The full-frame window coordinates and valid-area fractions are saved per site.')
    s=s.split('<!-- measured-qualification -->')[0]
    s=re.sub(r'<section class="notice"><h2>Measured qualification</h2><pre>.*?</pre></section>', '', s, flags=re.S)
    if s.endswith('</body></html>'):s=s[:-14]
    block='<section class="notice"><h2>Measured qualification</h2><pre>'+html.escape(reading)+'</pre></section>'
    s=s.replace('<h2>Exact methods, measurements and qualification</h2>',block+'<h2>Exact methods, measurements and qualification</h2>',1)
    report.write_text(s+'<!-- measured-qualification --></body></html>')
    source=Path(__file__)
    (OUT/'provenance'/source.name).write_bytes(source.read_bytes())
    manifest['presentation_sources']={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest['presentation_note']='Post-run qualification/caption correction from saved tables; extraction and frozen choices unchanged. Rebuild after run_audit with finalize_report.'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Measured qualification added without re-extraction or changing the fixed bank.')


if __name__=='__main__':main()
