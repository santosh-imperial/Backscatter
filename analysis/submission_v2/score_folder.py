"""Score incoming detector sets with the frozen D52 material-family model; fit nothing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
import tifffile

from .runtime import HERE, ROOT, MODEL_VERSION, PRIMARY_FAMILY, empty_output, load_runtime, sha, write_json


def explain_pairwise(model, x, site, assigned, runner, known, flags):
    z=model.pipeline['scale'].transform(model.pipeline['impute'].transform(x[None,:]))[0]
    a,b=model.classes.index(assigned),model.classes.index(runner)
    lr=model.pipeline['lr']
    contributions=(lr.coef_[a]-lr.coef_[b])*z
    intercept=float(lr.intercept_[a]-lr.intercept_[b])
    logits=model.pipeline.decision_function(x[None,:])[0]
    if not np.isclose(intercept+contributions.sum(),logits[a]-logits[b],rtol=1e-10,atol=1e-10):
        raise ValueError('Pairwise logit explanation does not reproduce model decision')
    texture={'etd_crack_density_graphite','crack_g_0.05','crack_g_0.1','crack_g_0.2','crack_g_0.3','ridge_p97',
             'inlens_particle_texture','inlens_particle_texture_p90','inlens_speckled_particle_frac','inlens_grad_energy'}
    rows=[]
    for j in np.argsort(-np.abs(contributions),kind='stable'):
        feature=model.features[j]
        mask_sensitive=feature.startswith(('bright_','inlens_particle_')) or feature in {'inlens_speckled_particle_frac','etd_crack_density_particles'}
        quality='unreliable bright mask / interior selection' if 'bright_low_contrast' in flags and mask_sensitive else ''
        rows.append(dict(sample_id=site,assigned_batch=assigned,runner_up=runner,feature=feature,
                         family='acquisition-sensitive appearance' if feature in texture else 'morphology',
                         value=x[j],measurement_missing_imputed=bool(np.isnan(x[j])),quality=quality,
                         contribution=float(contributions[j]),intercept_difference=intercept,
                         **{f'median_{c}':known.loc[known.batch==c,feature].median() for c in model.classes}))
    return rows


def inspect_input(folder):
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise ValueError(f"No TIFFs found in {folder}")
    rows = []
    for path in files:
        match = re.fullmatch(r"img_(\w+)_(BSE|ETD|SE|Inlens)\.tif", path.name)
        if not match:
            raise ValueError(f"Unsupported filename: {path.name}")
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            row = dict(file=path.name, site=match[1], detector="ETD" if match[2] == "SE" else match[2],
                       shape=list(series.shape), dtype=str(series.dtype), sha256=sha(path))
        rows.append(row)
    for site in sorted({r["site"] for r in rows}):
        group = [r for r in rows if r["site"] == site]
        if len(group) != 3 or {r["detector"] for r in group} != {"BSE", "ETD", "Inlens"}:
            raise ValueError(f"{site}: expected exactly one BSE, ETD/SE and Inlens image")
        if len({tuple(r["shape"]) for r in group}) != 1 or any(r["dtype"] != "uint8" for r in group):
            raise ValueError(f"{site}: expected matching uint8 detector geometry")
        shape = group[0]["shape"]
        if not (len(shape) == 2 or (len(shape) == 3 and shape[-1] == 3)):
            raise ValueError(f"{site}: expected a grayscale or RGB image, got shape {shape}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Hackathon-Polaron-test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=HERE / "freeze")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "_scratch/submission_v2_features")
    args = parser.parse_args(argv)
    empty_output(args.out)
    if args.model_version != MODEL_VERSION:
        raise ValueError("Declared model version does not match this runner")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    snapshot, CAT = load_runtime(args.freeze)
    files = inspect_input(args.input)
    training_hashes = {r['sha256'] for r in snapshot['raw_training_manifest']}
    if any(r['sha256'] in training_hashes for r in files):
        raise ValueError('Incoming TIFF duplicates a raw training file; do not score it as a holdout')
    saved = joblib.load(args.freeze / "models.joblib")  # trusted local known-only artifact
    known, models = saved["known"], saved["models"]
    if {r["site"] for r in files} & set(known.site):
        raise ValueError("Incoming site IDs overlap training IDs; do not score as a holdout")
    if saved["models"][PRIMARY_FAMILY].features != snapshot["primary_features"]:
        raise ValueError("Saved primary feature list does not match declared snapshot")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {len(files)//3} sites with frozen known-only models; extracting incoming features...", flush=True)
    sites = CAT.load_known([str(args.input.resolve())], cache_dir=str(args.cache_dir.resolve()), verbose=False)
    table = CAT.categorise_sites(sites, models, saved["ood_models"], role="scored", top=10)
    sites.to_csv(args.out / "features.csv", index=False)
    table.to_csv(args.out / "site_table.csv", index=False)
    primary = models[PRIMARY_FAMILY]
    X = sites[primary.features].to_numpy(float)
    deletion_scores = np.stack([CAT._proba_full(pipe, X, len(primary.classes))
                               for pipe in saved["primary_deletion_models"]])
    records, details = [], {}
    for i, row in table.iterrows():
        probabilities = np.array([row[f"mp_{c}__{PRIMARY_FAMILY}"] for c in primary.classes])
        winner = int(np.argmax(probabilities)); order = np.argsort(-probabilities, kind="stable")
        assignment = primary.classes[winner]
        counts = {c: int((deletion_scores[:, i].argmax(axis=1) == j).sum()) for j, c in enumerate(primary.classes)}
        flags = [f for f in ("bright_low_contrast", "grey_pore", "raised_black_level", "contrast_stretched_bse")
                 if pd.notna(row.get(f)) and bool(row.get(f))]
        contributions = CAT.categoriser_contributions(primary.pipeline, X[i], primary.features, winner,
                                                     primary.ref_median, primary.ref_mad, top=len(primary.features))
        for item in contributions:
            item["family"] = ("morphology" if item["feature"] in CAT.MORPH_FEATURES
                              else "acquisition-sensitive texture" if item["feature"] in CAT.TEXTURE_FEATURES
                              else "acquisition")
            item["measurement_missing_imputed"] = bool(np.isnan(X[i, primary.features.index(item["feature"])]))
        note = ("Uncalibrated model scores; deletion counts measure training-set sensitivity, not independent votes "
                "or probabilities of correctness. True batch label pending.")
        record = dict(sample_id=row.site, predicted_batch=assignment,
                      **{f"model_score_{c}": float(probabilities[j]) for j, c in enumerate(primary.classes)},
                      top_model_score=float(probabilities[winner]),
                      runner_up=primary.classes[int(order[1])], margin=float(probabilities[order[0]] - probabilities[order[1]]),
                      deletion_same_assignment=counts[assignment], deletion_fit_count=len(deletion_scores),
                      **{f"deletion_votes_{c}": n for c, n in counts.items()},
                      assigned_score_deletion_min=float(deletion_scores[:, i, winner].min()),
                      assigned_score_deletion_max=float(deletion_scores[:, i, winner].max()),
                      morphology_only_bet=row.argmax__morph, acquisition_only_bet=row.argmax__acq,
                      baseline_morphology_percentile=float(row.ood_pct_knn__morph),
                      exceeds_observed_baseline_max=bool(row.ood_exceeds_knn__morph),
                      acquisition_flags="|".join(flags) or "none of listed flags",
                      explanation=row[f"cat_top_features__{PRIMARY_FAMILY}"], uncertainty_note=note)
        records.append(record)
        details[str(row.site)] = dict(contributions=contributions, deletion_counts=counts,
                                    missing_primary_features=[f for f, v in zip(primary.features, X[i]) if np.isnan(v)],
                                    baseline_nearest_sites=row.ood_nearest_ref__morph,
                                    baseline_drivers=row.ood_top_features__morph)
    submission = pd.DataFrame(records)
    submission.to_csv(args.out / "submission_sites.csv", index=False)
    image_rows = pd.DataFrame(files).rename(columns={"site": "sample_id", "file": "filename"})
    image_rows.merge(submission, on="sample_id", validate="many_to_one").to_csv(args.out / "submission_images.csv", index=False)
    contrasts=[]
    for i, row in submission.iterrows():
        contrasts.extend(explain_pairwise(primary, X[i], row.sample_id, row.predicted_batch, row.runner_up, known, row.acquisition_flags))
    pd.DataFrame(contrasts).to_csv(args.out / "driver_contrasts.csv", index=False)
    for name in snapshot["evaluation_sha256"]:
        (args.out / name).write_bytes((args.freeze / "evaluation" / name).read_bytes())
    write_json(args.out / "explanations.json", details)
    write_json(args.out / "input_manifest.json", files)
    for r in files:
        if sha(args.input / r["file"]) != r["sha256"]:
            raise ValueError("Input mutated during scoring; first outputs are not valid")
    receipt = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=time.monotonic()-t0, input_directory=str(args.input.resolve()),
                   source_snapshot_sha256=sha(args.freeze / "snapshot.json"),
                   snapshot_path=str((args.freeze / "snapshot.json").resolve()), model_version=MODEL_VERSION, primary_family=PRIMARY_FAMILY, model_sha256=sha(args.freeze / "models.joblib"),
                   scorer_sha256=sha(Path(__file__)), n_sites=len(sites), n_images=len(files),
                   training_only=True, first_drop_labels_pending=True, no_test_fitting_or_model_selection=True,
                   source_execution="preserved source snapshot under freeze/sources; live package edits ignored",
                   bet_rule="D52 material-family argmax; no comparator fallback or post-score overrides",
                   output_sha256={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    write_json(args.out / "prediction_receipt.json", receipt)
    print(submission[["sample_id", "predicted_batch", "top_model_score", "margin", "deletion_same_assignment",
                      "deletion_fit_count", "baseline_morphology_percentile", "acquisition_flags"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
