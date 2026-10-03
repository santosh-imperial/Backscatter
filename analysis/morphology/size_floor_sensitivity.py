"""E18 measurement audit: centroid spacing at three component area floors.

This supplements the pre-specified screen after inspecting its image overlays.
It does not select a floor for the QC pipeline or run new significance tests.
"""
import json

import numpy as np
import pandas as pd

from analysis.morphology.run_analysis import OUT, SEED, N_CSR, spacing_summary, median_interval, sha256


def main():
    sites = pd.read_csv(OUT / "morphology_sites.csv")
    components = pd.read_csv(OUT / "component_geometry.csv.gz")
    rows = []
    for i, r in enumerate(sites.itertuples()):
        objects = components[(components.site == r.site) & (components.phase == "bright")]
        for floor in (50, 500, 2000):
            sub = objects[objects.area >= floor]
            measurement = spacing_summary(sub, (r.H, r.W), seed=SEED + i)
            if floor == 50:
                assert np.isclose(measurement["bright_nn_csr_ratio"], r.bright_nn_csr_ratio, atol=1e-10)
            rows.append(dict(batch=r.batch, site=r.site, min_component_area_px=floor, n_components=len(sub),
                             bright_low_contrast=r.bright_low_contrast, grey_pore=r.grey_pore, **measurement))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "spacing_area_floor_sites.csv", index=False)
    comparisons = []
    for floor, sub in result.groupby("min_component_area_px"):
        sub = sub[~sub.bright_low_contrast & ~sub.grey_pore]
        for reference, batch in (("Batch_3", "Batch_1"), ("Batch_3", "Batch_2"), ("Batch_1", "Batch_2")):
            a = sub[sub.batch == reference].bright_nn_csr_ratio.dropna().to_numpy()
            b = sub[sub.batch == batch].bright_nn_csr_ratio.dropna().to_numpy()
            lo, hi = median_interval(a, b, SEED)
            comparisons.append(dict(min_component_area_px=floor, reference=reference, batch=batch,
                                    n_ref=len(a), n_batch=len(b), ref_median=np.median(a), batch_median=np.median(b),
                                    median_delta=np.median(b)-np.median(a), delta_ci_low=lo, delta_ci_high=hi))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "spacing_area_floor_comparisons.csv", index=False)
    protocol = dict(purpose="Post-screen image-driven measurement audit; no chosen floor or new hypothesis test",
                    floors_px=[50, 500, 2000], n_csr=N_CSR, seed=SEED,
                    inputs={name: sha256(OUT / name) for name in ("morphology_sites.csv", "component_geometry.csv.gz", "manifest.json")},
                    source_sha256=sha256(__file__))
    (OUT / "spacing_area_floor_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
