"""Append E43S numerical results, preserving every historical ledger byte."""
import csv
import hashlib
import io
import json
import numpy as np
import pandas as pd
from analysis.submission_v2.runtime import ROOT, sha, write_json
from .run import OUT


def main():
    path=ROOT/'experiments/registry.csv';original=path.read_bytes()
    assert not any(r['experiment']=='E43S' for r in csv.DictReader(io.StringIO(original.decode())))
    fields=next(csv.reader(io.StringIO(original.decode())));rows=[]
    def add(metric,value,candidate='',cohort='known_oof',reference='actual_frozen_v2',n=31,note=''):
        if not np.isfinite(value):return
        row={k:'' for k in fields}
        row.update(experiment='E43S',date='2026-10-04',metric=metric,reference=reference,batch=cohort,kpi=candidate,
            statistic='nested crop-held-out development/technical',n_ref=31,n_batch=n,value=value,
            unit='nats' if 'logloss' in metric or 'nll' in metric else 'count' if metric.startswith('n_') else 'score',
            note=('Unknown shared sources; original frozen v2, not anchor-ordered E42 comparator. No promotion; exact organiser payoff unknown. '+note).strip())
        rows.append(row)
    for _,r in pd.read_csv(OUT/'summary.csv').iterrows():
        for m in pd.read_csv(OUT/'summary.csv').select_dtypes(include='number'):
            add(m,r[m],r.candidate,r.cohort,n=r.n_sites)
    for file,prefix in [('threshold_counts.csv','threshold'),('reliability_bins.csv','reliability')]:
        df=pd.read_csv(OUT/file)
        for _,r in df.iterrows():
            ref=f'threshold={r.threshold}' if prefix=='threshold' else f'bin=[{r.bin_low},{r.bin_high}]'
            for m in df.select_dtypes(include='number'):
                if m in ['threshold','bin_low','bin_high']:continue
                add(prefix+'_'+m,r[m],r.candidate,r.cohort,ref,n=31 if r.cohort=='known_oof' else 3,
                    note='Descriptive fixed thresholds/bins; not organiser cutoff or calibration proof.')
    for _,r in pd.read_csv(OUT/'temperatures.csv').iterrows():
        for m in ['alpha','temperature','training_balanced_nll','identity_training_balanced_nll']:
            add(m,r[m],'calibrated','scalar_fit',f'fold={r.fold}',int(r.n_calibration),
                'Fit loss is not held-out evidence; full scalar is deployment-only.')
    for _,r in pd.read_csv(OUT/'feedback_predictions.csv').iterrows():
        for m in ['score_Batch_1','score_Batch_2','score_Batch_3','bet_score']:
            add('revealed_'+m,r[m],r.candidate,'feedback_development',r.site,1,'Already revealed; no fitting/selection.')
    for m,v in json.loads((OUT/'validation.json').read_text()).items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):add(m,v,cohort='technical',n='')
    for m,v in json.loads((OUT/'test_receipt.json').read_text()).items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):add(m,v,cohort='technical',n='')
    buf=io.StringIO(newline='');csv.DictWriter(buf,fieldnames=fields,lineterminator='\n').writerows(rows)
    with path.open('ab') as stream:stream.write(buf.getvalue().encode())
    assert path.read_bytes()[:len(original)]==original
    write_json(OUT/'registry_append_receipt.json',dict(experiment='E43S',appended_rows=len(rows),historical_prefix_bytes=len(original),
        historical_prefix_sha256=hashlib.sha256(original).hexdigest(),current_sha256=sha(path)))
    print('Appended',len(rows),'E43S rows; historical bytes unchanged.')


if __name__=='__main__':main()
