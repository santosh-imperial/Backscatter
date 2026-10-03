"""E25 known-batch integration check using explicitly reused primary caches.

Run after the three analyses and feature-regression tests. Existing primary,
image, particle and patch tables are copied byte-for-value, with independently
recomputed E25 secondary columns added. This avoids repeating unchanged ETD and
texture extraction; the normal extractor computes everything for a new folder.
No historical cache file is overwritten. Provenance records this warm replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from polaron_qc import PRIMARY_KPIS
from polaron_qc import battery_metrics, features, report, secondary, void_metrics

OUT = ROOT / "analysis/battery/output"
CACHE = ROOT / "analysis_cache/features"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage():
    npath, vpath = OUT/"neighbourhood_sites.csv", OUT/"void_sites.csv"
    neighbourhood = pd.read_csv(npath).set_index(["batch", "site"])
    voids = pd.read_csv(vpath)
    assert len(voids)==93 and voids.groupby(["batch","site"]).size().eq(3).all()
    assert len(neighbourhood)==31 and neighbourhood.index.is_unique
    nominal_voids = voids[voids.threshold_offset==0].set_index(["batch","site"])
    assert nominal_voids.index.is_unique and set(nominal_voids.index)==set(neighbourhood.index)
    provenance = dict(experiment="E25", feature_version=features.FEATURE_VERSION,
                      mode="verified_primary_cache_reuse_plus_recomputed_secondary_masks",
                      secondary_version=secondary.VERSION, source_tables={str(p.relative_to(ROOT)):digest(p) for p in (npath,vpath)},
                      batches={}, claims_not_made=["full cold extraction of every channel repeated", "expert segmentation accuracy", "unseen-batch validation"])
    for batch in ("Batch_1","Batch_2","Batch_3"):
        options = []
        for p in CACHE.glob(batch+"_*.sites.parquet"):
            df = pd.read_parquet(p)
            if "battery_secondary_version" not in df: options.append((p,df))
        if len(options)!=1:
            raise RuntimeError(f"Expected one historical feature-cache family for {batch}, found {len(options)}; use cold normal extraction instead")
        old_path, old = options[0]
        old = old.set_index(["batch","site"])
        ns, vs = neighbourhood.loc[old.index], nominal_voids.loc[old.index]
        for k in ("th_lo","th_hi"):
            np.testing.assert_allclose(old[k],ns[k],rtol=0,atol=1e-10)
        np.testing.assert_allclose(old.H,vs.H_trim,rtol=0,atol=0)
        np.testing.assert_allclose(old.W,vs.W,rtol=0,atol=0)
        extra = {}
        for k in list(battery_metrics.KPI_NAMES)+list(battery_metrics.DIAGNOSTIC_NAMES):
            extra[k] = ns[k]
        for k in battery_metrics.KPI_NAMES:
            for suffix in ("_sensitivity_min","_sensitivity_max","_sensitivity_n"):
                extra[k+suffix] = ns[k+suffix]
        vv=voids[voids.batch==batch].set_index("site")
        for k in void_metrics.KPI_NAMES+void_metrics.DIAGNOSTIC_NAMES:
            extra[k]=vs[k]
        for k in void_metrics.KPI_NAMES:
            band=vv.groupby(level=0)[k].agg(["min","max","count"])
            for suffix,col in (("_sensitivity_min","min"),("_sensitivity_max","max"),("_sensitivity_n","count")):
                extra[k+suffix] = pd.Series(band.loc[old.index.get_level_values("site"),col].to_numpy(),index=old.index)
        merged = old.copy()
        for k,v in extra.items():
            assert k not in old, (k,"overlaps historical measurements")
            merged[k]=v
        merged["battery_secondary_version"] = secondary.VERSION
        pd.testing.assert_frame_equal(merged[old.columns],old)
        folder=ROOT/"Dataset"/batch
        current_hash=features.folder_hash(folder)
        paths=features._cache_paths(str(CACHE),batch,current_hash,"parquet")
        old_prefix=str(old_path).removesuffix(".sites.parquet")
        source_hashes={}
        for name,p in paths.items():
            source=Path(old_prefix+"."+name+".parquet")
            table=merged.reset_index() if name=="sites" else pd.read_parquet(source)
            source_hashes[str(source.relative_to(ROOT))]=digest(source)
            if Path(p)==source: raise RuntimeError("Refusing to overwrite historical cache")
            table.to_parquet(p,index=False)
        # Cache-loading API sees all new columns, while historical values match.
        loaded=features.extract_batch(folder,cache_dir=str(CACHE),verbose=False)["sites"].set_index(["batch","site"])
        pd.testing.assert_frame_equal(loaded[old.columns],old)
        for k in secondary.EXTRACTED_NAMES:
            np.testing.assert_allclose(loaded[k],merged[k],rtol=0,atol=1e-12,equal_nan=True)
        provenance["batches"][batch]=dict(n_sites=len(old),source_sha256=source_hashes,
                                           output_paths={k:str(Path(p).relative_to(ROOT)) for k,p in paths.items()},
                                           original_columns_unchanged=True)
    (OUT/"pipeline_cache_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--no-reports",action="store_true");args=parser.parse_args()
    provenance=stage()
    if args.no_reports:return
    summaries=[]; results=[]
    for batch in ("Batch_1","Batch_2"):
        result=report.build_result(str(ROOT/"Dataset/Batch_3"),str(ROOT/"Dataset"/batch),cache_dir=str(CACHE),verbose=True)
        assert result["meta"]["primary_kpis"]==PRIMARY_KPIS
        assert set(result["compare"].kpi).isdisjoint(secondary.EXTRACTED_NAMES)
        assert set(result["c2st_material"]["kpis"]).isdisjoint(secondary.EXTRACTED_NAMES)
        result["meta"]["notes"].append("E25 warm replay: unchanged verified primary/image/particle/patch caches reused; secondary masks independently recomputed. See analysis/battery/output/pipeline_cache_provenance.json.")
        path=ROOT/"reports"/f"qc_{batch}.html"
        report.render_report(result,str(path))
        html=path.read_text()
        assert "Battery geometry candidates" in html and "not an intact-particle share" not in html  # helper-only text is not a production intactness score
        assert "more favourable macro-pore transport" not in html and "violated by plate alignment" not in html
        assert "bright_void_boundary_frac" not in html
        s=result["battery_secondary"].copy();s.insert(0,"batch",batch);s.insert(0,"reference","Batch_3");summaries.append(s)
        result["compare"].to_csv(OUT/f"pipeline_{batch}_primary_secondary_comparisons.csv",index=False)
        results.append(dict(batch=batch,verdict=result["verdict"]["verdict"],decision_stability=result["stability"]["share"],
                            primary_kpis=result["meta"]["primary_kpis"],secondary_geometry_kpis=list(secondary.KPI_NAMES),
                            threshold_hash=result["meta"]["thresholds_hash"],report=str(path.relative_to(ROOT)),
                            runtime_s=result["meta"]["runtime_s"]))
    pd.concat(summaries).to_csv(OUT/"pipeline_battery_summary.csv",index=False)
    (OUT/"pipeline_verification.json").write_text(json.dumps(dict(experiment="E25",cache_mode=provenance["mode"],results=results),indent=2)+"\n")
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
