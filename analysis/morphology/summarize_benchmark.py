"""Measurement-sensitivity diagnostics, not expert accuracy or batch verdicts."""
import json

import numpy as np
import pandas as pd

from analysis.morphology.build_benchmark import OUT,WIDTH_KEYS,write_json


def main():
    width=pd.read_csv(OUT/'local_width_sites.csv')
    nominal=width[width.threshold_offset==0]
    band=pd.read_csv(OUT/'local_width_sensitivity.csv')
    hyst=pd.read_csv(OUT/'hysteresis_sites.csv')
    conn=pd.read_csv(OUT/'connectivity_sites.csv')
    rows=[]
    def record(metric,values,units,scope,note):
        values=pd.Series(values).dropna()
        rows.append(dict(metric=metric,scope=scope,n_sites=len(values),median=float(values.median()) if len(values) else np.nan,
                         min=float(values.min()) if len(values) else np.nan,max=float(values.max()) if len(values) else np.nan,
                         units=units,note=note))
    for scope,w in [('all_known_sites',nominal),('pore_mode_quality_usable',nominal[~nominal.grey_pore])]:
        for k in WIDTH_KEYS:
            record(k,w[k],'px',scope,'Descriptive full-site centreline width; edge components excluded; no expert accuracy claim.')
            b=band[(band.kpi==k)&band.site.isin(w.site)]
            record(k+'_relative_threshold_band',(b.method_band_hi-b.method_band_lo)/b.nominal,'fraction',scope,
                   '(maximum - minimum across ±5 grey levels) / nominal; method sensitivity, not a confidence interval.')
    for scope,h in [('all_known_sites',hyst),('bright_quality_usable',hyst[~hyst.bright_low_contrast])]:
        for k in ['agreement_iou','changed_image_fraction','changed_bright_area_relative','delta_frac','delta_d50']:
            record('hysteresis_'+k,h[k],'px' if k.endswith('d50') else 'fraction',scope,
                   'Agreement/change between algorithms, not segmentation accuracy; no ground-truth labels yet.')
    for phase in ['pore','bright']:
        c=conn[conn.phase==phase]
        record(phase+'_connectivity_count_relative_delta',(c.count4_per_Mpx-c.count8_per_Mpx)/c.count8_per_Mpx,'fraction','all_known_sites',
               '4-connected vs existing 8-connected component density; same binary mask, different object definition.')
        record(phase+'_connectivity_d50_delta',c.d50_4_px-c.d50_8_px,'px','all_known_sites',
               'Connectivity sensitivity only; existing production definition remains 8-connected.')
    pd.DataFrame(rows).to_csv(OUT/'measurement_diagnostics.csv',index=False)
    manifest=json.loads((OUT/'roi_manifest.json').read_text())
    status=dict(experiment='E24',n_sites=len(nominal),n_threshold_variants=len(width),
                n_roi_sites=len(manifest['rois']),n_development_roi_sites=sum(r['split']=='development' for r in manifest['rois']),
                n_held_out_roi_sites=sum(r['split']=='held_out_site' for r in manifest['rois']),
                n_expert_reviewed_rois=0,expert_accuracy_available=False,
                n_nominal_sites_with_crack_width=int(nominal.crack_local_width_d50_px.notna().sum()),
                interpretation='Synthetic geometry and output-contract checks passed. Segmentation/width accuracy and manufacturing meaning await expert annotations. Exclusion of full-image-edge components can remove the largest visible voids. Hysteresis does not restore missing phase contrast.',
                next_step='Annotate development and held-out-site ROIs separately; evaluate manual errors, preserve explicit unmeasurable/uncertain cases; no primary promotion.')
    write_json(OUT/'status.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
