"""Post-run interpretation of saved K values; no extraction or settings changes."""
from pathlib import Path
import hashlib
import html
import json
import pandas as pd

OUT=Path(__file__).resolve().parent

def main():
    comp=pd.read_csv(OUT/'comparisons.csv');controls=pd.read_csv(OUT/'control_predictions.csv')
    sens=pd.read_csv(OUT/'sensitivity_summary.csv')
    def row(k,mode='nominal',view='quality_matched'):
        return comp[comp.kpi.eq(k)&comp['mode'].eq(mode)&comp.view.eq(view)&comp.reference.eq('Batch_3')&comp.batch.eq('Batch_1')].iloc[0]
    solidity='bright_solidity_count_q10';phase='bright_pore_crosscorr_xy_contrast_256px'
    a,b=row(solidity),row(phase)
    r2=controls[controls.kpi.eq(solidity)&controls.view.eq('quality_matched')&controls.control.eq('existing_geometry_loading')].loo_r2.iloc[0]
    lines=['# Measured qualification — task K / E30K','',
        'Two correlated nominal quality-matched comparisons have intervals excluding zero. Both remain exploratory and outside QC; neither establishes a manufacturing defect, a battery mechanism or acceptance/rejection. Sites may share specimens, and phase masks/image direction are unreviewed.','',
        '## Solidity lower tail: sensitive and partly predictable from existing geometry','',
        f'Batch 1 − Batch 3 is {a.median_difference:.6f}, site-bootstrap 95% [{a.ci_low:.6f}, {a.ci_high:.6f}], at {int(a.n_batch)} incoming / {int(a.n_reference)} reference sites. The nominal grey-pore-included interval spans zero. Each of the ±5 threshold and 100 px² floor quality-matched intervals also spans zero. The existing geometry/loading LOO control reaches R² {r2:.3f}. This is a count-weighted tail, distinct from the existing area-weighted mean; it is not established incremental material information.','',
        'Review whether low-solidity components are physical sections, particle rims, threshold fragments or preparation effects. A lower raster solidity can reflect any of those. Preserve uncertain/unmeasurable cases; independent annotations are needed.','',
        '## 256 px bright/void direction contrast: repeatable within this limited mask test','',
        f'Batch 1 − Batch 3 is {b.median_difference:.6f}, site-bootstrap 95% [{b.ci_low:.6f}, {b.ci_high:.6f}], at {int(b.n_batch)} incoming / {int(b.n_reference)} reference sites. The nominal, −5 and +5 quality-matched intervals exclude zero with the same direction, as do the corresponding ordinary-reference sensitivity intervals. The floor100 raster duplicate is identical by definition and adds no robustness evidence.','',
        'This supports further image/phase/section review of a finite-frame arrangement difference, not an established material or failure mechanism. Only two fixed lags were evaluated. Removing known long-void reference sites changes the reference population; it does not certify a clean baseline. Marginal normalisation and poor linear control predictability do not prove acquisition invariance.','',
        'Review bright/void masks at the relevant image-axis spacing, the direction/context of each section and possible preparation/session fingerprints. Confirm specimen independence and repeat sections before a material interpretation. No new tuning, extra lag, trained classifier or unseen-batch selection was run.','',
        'Other nominal quality-matched intervals include zero; that is limited evidence at these usable n, not equivalence. These are unadjusted correlated exploratory intervals. All seven material/QC uses stay deferred.','']
    text='\n'.join(lines).rstrip()+'\n'
    (OUT/'review.md').write_text(text)
    p=OUT/'report.html';page=p.read_text();assert '<h2>Measured qualification</h2>' not in page
    block='<h2>Measured qualification</h2><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre>'
    page=page.replace('<h2>Measured site distributions</h2>',block+'<h2>Measured site distributions</h2>',1);p.write_text(page)
    p=OUT/'findings.md';p.write_text(p.read_text().rstrip()+'\n\n## Post-run qualification\n\n'+text.split('\n',1)[1].lstrip())
    snap=OUT/'provenance/qualify_report.py';snap.write_bytes(Path(__file__).read_bytes())
    (OUT/'presentation_manifest.json').write_text(json.dumps(dict(source='qualify_report.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Post-run qualification of saved tables only; extraction/settings unchanged'),indent=2)+'\n')
    print('Qualified both apparent nominal findings from saved sensitivity/control rows; no re-extraction.')

if __name__=='__main__':main()
