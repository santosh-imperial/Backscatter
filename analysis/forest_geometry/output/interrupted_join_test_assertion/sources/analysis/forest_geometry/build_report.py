"""Render the fixed E41S forest/geometry diagnostics; never fit or select a model."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output/evaluation"
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
MAIN = ["base_l1", "base_rf", "geometry_l1", "geometry_rf"]
ORDER = ["legacy_l1", *MAIN, "quality_rf", "availability_rf"]
LABELS = {
    "legacy_l1": "Ungated L1 · frozen-procedure reference",
    "base_l1": "Gated L1 · existing 29",
    "base_rf": "Gated forest · existing 29",
    "geometry_l1": "Gated L1 · existing 29 + geometry 8",
    "geometry_rf": "Gated forest · existing 29 + geometry 8",
    "quality_rf": "Quality-flags forest · diagnostic control",
    "availability_rf": "Availability-pattern forest · diagnostic control",
}
SHORT = {
    "legacy_l1": "Ungated L1",
    "base_l1": "Base L1",
    "base_rf": "Base forest",
    "geometry_l1": "Geometry L1",
    "geometry_rf": "Geometry forest",
    "quality_rf": "Quality control",
    "availability_rf": "Availability control",
}
COLORS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}


def embed(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data", escape=True,
                         float_format=lambda value: f"{value:.3f}")


def read_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name)
    if frame.empty:
        raise ValueError(f"Report input is empty: {name}")
    return frame


def validate_inputs(summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Fail closed if candidate or crop coverage cannot support the shown comparison."""
    if summary.candidate.duplicated().any() or set(summary.candidate) != set(ORDER):
        raise ValueError("The report requires the seven fixed candidate summaries")
    if set(predictions.candidate) != set(ORDER):
        raise ValueError("Out-of-fold predictions do not cover all fixed candidates")
    shared_sites = None
    for name in ORDER:
        rows = predictions.loc[predictions.candidate.eq(name)]
        if len(rows) != 31 or rows.site.duplicated().any():
            raise ValueError(f"Expected one withheld prediction for each of 31 crops: {name}")
        counts = rows.true_batch.value_counts().to_dict()
        if counts != {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}:
            raise ValueError(f"Unexpected known-label counts: {name}")
        identity = set(zip(rows.site, rows.true_batch))
        if shared_sites is not None and identity != shared_sites:
            raise ValueError("Candidate rows do not share the same held-out crops/labels")
        shared_sites = identity
        scores = rows[["score_" + c for c in CLASSES]].to_numpy(float)
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1):
            raise ValueError(f"Invalid class scores: {name}")
        chosen = np.asarray(CLASSES)[scores.argmax(axis=1)]
        if not np.array_equal(chosen, rows.predicted_batch.to_numpy()):
            raise ValueError(f"Prediction and displayed score mapping disagree: {name}")
        correct = chosen == rows.true_batch.to_numpy()
        if not np.array_equal(correct, rows.correct.to_numpy()):
            raise ValueError(f"Incorrect stored correctness flags: {name}")
        result = summary.loc[summary.candidate.eq(name)].iloc[0]
        recalls = [float(correct[rows.true_batch.eq(c)].mean()) for c in CLASSES]
        if int(result.n_sites) != len(rows) or int(result.n_correct) != int(correct.sum()):
            raise ValueError(f"Summary crop/correct counts disagree with saved scores: {name}")
        if not np.allclose(recalls, [result["recall_" + c] for c in CLASSES]):
            raise ValueError(f"Summary class recalls disagree with saved scores: {name}")
        if not np.isclose(np.mean(recalls), result.balanced_accuracy):
            raise ValueError(f"Summary balanced accuracy disagrees with saved scores: {name}")


def comparison_plot(summary: pd.DataFrame) -> Path:
    rows = summary.set_index("candidate").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    x = np.arange(len(ORDER))
    palette = ["#8b9ba6", "#2a78d6", "#457689", "#d38a38", "#eb6834", "#adb7bc", "#adb7bc"]
    for ax, metric, title in zip(axes, ["balanced_accuracy", "balanced_logloss"],
                                 ["Balanced accuracy · higher", "Balanced log loss · lower"]):
        values = rows[metric].to_numpy(float)
        ax.bar(x, values, color=palette)
        ax.set_xticks(x, [SHORT[n] for n in ORDER], rotation=34, ha="right", fontsize=9)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel("Crop-held-out development diagnostic")
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        if metric == "balanced_accuracy":
            ax.axhline(1 / 3, ls="--", color="#596e7a", lw=1, label="Uniform-bet reference")
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.set_ylim(0, max(values.max() * 1.17, 1.2))
    fig.suptitle("Seven fixed diagnostics · parent-image leakage remains unresolved", fontsize=13)
    fig.tight_layout()
    path = OUT / "plots_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_plot(predictions: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.flat, MAIN):
        rows = predictions.loc[predictions.candidate.eq(name)]
        matrix = pd.crosstab(rows.true_batch, rows.predicted_batch).reindex(
            index=CLASSES, columns=CLASSES, fill_value=0).to_numpy(int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=17)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > 8 else "#17364a", fontsize=13)
        ax.set_xticks(range(3), ["B1", "B2", "B3"])
        ax.set_yticks(range(3), ["B1 · 7 crops", "B2 · 7 crops", "B3 · 17 crops"])
        ax.set_xlabel("Predicted visual group")
        ax.set_ylabel("Supplied visual group")
        ax.set_title(SHORT[name], loc="left", fontsize=12)
    fig.suptitle("Fixed 2 × 2 comparison · each count is a crop", fontsize=14)
    fig.tight_layout()
    path = OUT / "plots_confusions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def recall_plot(summary: pd.DataFrame) -> Path:
    rows = summary.set_index("candidate").loc[MAIN]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(MAIN))
    for index, cls in enumerate(CLASSES):
        values = rows["recall_" + cls].to_numpy(float)
        ax.bar(x + (index - 1) * .24, values, width=.23, color=COLORS[cls],
               label=cls.replace("_", " ") + (" · 17 crops" if cls == "Batch_3" else " · 7 crops"))
    ax.set_xticks(x, [SHORT[n] for n in MAIN])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Recall among supplied crops")
    ax.set_title("Class tradeoffs remain visible", loc="left", fontsize=13)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.2), fontsize=9)
    fig.tight_layout()
    path = OUT / "plots_recalls.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.set_index("candidate").loc[ORDER].reset_index()
    frame["Correct / crops"] = frame.n_correct.astype(int).astype(str) + " / " + frame.n_sites.astype(int).astype(str)
    frame.candidate = frame.candidate.map(LABELS)
    columns = ["candidate", "n_features", "Correct / crops", "balanced_accuracy",
               *["recall_" + c for c in CLASSES], "balanced_logloss", "balanced_brier"]
    return frame[columns].rename(columns={"candidate": "Fixed candidate", "n_features": "Inputs",
        "balanced_accuracy": "Balanced accuracy", "balanced_logloss": "Balanced log loss",
        "balanced_brier": "Balanced Brier", **{"recall_" + c: "Recall " + c.replace("_", " ") for c in CLASSES}})


def _nuisance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["candidate", "variant"], sort=False).agg(
        crops=("site", "size"), changed_bets=("bet_changed", "sum"),
        median_largest_score_change=("max_absolute_score_change", "median"),
        largest_score_change=("max_absolute_score_change", "max")).reset_index()


def _feedback_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ORDER:
        for _, row in frame.loc[frame.candidate.eq(name)].iterrows():
            rows.append({"Fixed candidate": LABELS[name], "Crop": row.site,
                "Truth": row.true_batch.replace("_", " "), "Bet": row.predicted_batch.replace("_", " "),
                "Bet score": row["score_" + row.predicted_batch],
                "True-class score": row["score_" + row.true_batch], "Correct": row.correct,
                "Observed inputs": row.n_observed_inputs})
    return pd.DataFrame(rows)


def _image_references(receipt: dict, audit: pd.DataFrame) -> str:
    paths = receipt["examples"]
    figures = []
    for entry in paths:
        entry = {"path": entry} if isinstance(entry, str) else entry
        path = ROOT / entry["path"]
        if not path.is_file():
            raise ValueError(f"Missing actual-image reference: {path}")
        if "sha256" not in entry or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Actual-image reference identity mismatch: {path}")
        caption = entry.get("caption", f"{entry.get('batch', '')} / {entry.get('site', '')} · "
            f"display crop raw coordinates {entry.get('box_raw_yxyx', '')}; values measured on trimmed full field. "
            f"Expert validation: {entry.get('expert_validation', 'unreviewed')}.")
        rows = audit.loc[audit.site.eq(entry["site"]), ["feature", "raw_value", "observed_model_input", "reason"]]
        if len(rows) != 8:
            raise ValueError(f"Missing nominal per-feature eligibility for actual example: {entry['site']}")
        rows = rows.rename(columns={"raw_value": "Raw full-field value",
            "observed_model_input": "Observed model input", "reason": "Unavailability reason"})
        figures.append(f'<figure><img src="{embed(path)}" alt="Actual BSE image with measured geometry overlays">'
                       f'<figcaption>{html.escape(str(caption))}</figcaption></figure>'
                       f'<details><summary>Full-field values and eligibility · {html.escape(entry["site"])}</summary>'
                       f'<div class="scroll">{table(rows)}</div></details>')
    if not figures:
        raise ValueError("No actual-image geometry references in extraction receipt")
    return "".join(figures)


def main() -> None:
    summary = read_csv("known_summary.csv")
    predictions = read_csv("known_oof_predictions.csv")
    validate_inputs(summary, predictions)
    paired = read_csv("paired_comparisons.csv")
    feedback = read_csv("feedback_development_predictions.csv")
    sensitivity = read_csv("geometry_oof_sensitivity.csv")
    seed_sensitivity = read_csv("seed_sensitivity.csv")
    replacements = read_csv("group_replacement.csv")
    explanations = read_csv("explanations.csv")
    receipt = json.loads((HERE / "output/geometry_examples.json").read_text())
    from .geometry import FEATURES, DEFINITIONS, BRIGHT, PORE, JOINT
    definitions = pd.DataFrame([{"Feature": feature, "Fixed definition": DEFINITIONS[feature],
        "Mask dependency": "Bright + void" if feature in JOINT else "Bright" if feature in BRIGHT else "Void"}
        for feature in FEATURES])
    figures = [comparison_plot(summary), confusion_plot(predictions), recall_plot(summary)]
    lookup = summary.set_index("candidate")
    delta_rf = lookup.loc["base_rf", "balanced_accuracy"] - lookup.loc["base_l1", "balanced_accuracy"]
    delta_geometry = lookup.loc["geometry_rf", "balanced_accuracy"] - lookup.loc["base_rf", "balanced_accuracy"]
    delta_combined = lookup.loc["geometry_rf", "balanced_accuracy"] - lookup.loc["base_l1", "balanced_accuracy"]
    headline = (f"At the fixed bounds, changing gated L1 to the base forest changes balanced accuracy by "
                f"{delta_rf:+.3f}; adding geometry to that forest changes it by {delta_geometry:+.3f}. "
                f"The declared combined candidate differs from gated L1 by {delta_combined:+.3f}.")
    all_scores = predictions.loc[predictions.candidate.eq("geometry_rf")].copy()
    all_scores["bet_score"] = [row["score_" + row.predicted_batch] for _, row in all_scores.iterrows()]
    failure_table = all_scores.loc[all_scores.correct.eq(False), ["site", "true_batch", "predicted_batch", "bet_score", "n_observed_inputs"]]
    # Explanation units differ by model; keep the original table explicit and unsorted across models.
    explanation_cols = [c for c in ["cohort", "site", "predicted_batch", "comparison_batch", "feature",
                        "feature_group", "contribution", "observed", "rank", "explanation_type"] if c in explanations]
    chosen_explanations = explanations.loc[explanations.candidate.eq("geometry_rf")]
    if "rank" in chosen_explanations:
        chosen_explanations = chosen_explanations.loc[chosen_explanations["rank"].le(5)]
    coverage = predictions.groupby("candidate", sort=False).agg(
        crops=("site", "size"), min_observed=("n_observed_inputs", "min"),
        median_observed=("n_observed_inputs", "median"), max_observed=("n_observed_inputs", "max")).reset_index()
    margins = explanations.loc[explanations.candidate.eq("geometry_rf"),
        ["cohort", "site", "predicted_batch", "comparison_batch", "root_or_intercept_margin", "total_margin"]].drop_duplicates()
    geometry_audit = pd.read_csv(HERE / "output/geometry_known_availability.csv")
    geometry_audit = geometry_audit.loc[geometry_audit.variant.eq("nominal")]
    geometry_coverage = geometry_audit.groupby("feature", sort=False).agg(
        crops=("site", "size"), raw_finite=("raw_finite", "sum"),
        quality_eligible=("policy_eligible", "sum"), observed_inputs=("observed_model_input", "sum")).reset_index()
    diagnostics_html = ("<h3>Forest seed sensitivity</h3><p>Seeds 1 and 2 repeat the declared primary as a "
        "fixed stochastic check. Seed 0 remains primary; no seed is selected and no additional independent "
        "crops are created.</p><details><summary>Show all seed diagnostics</summary><div class=\"scroll\">" +
        table(seed_sensitivity) + "</div></details><h3>Fixed-model input-family replacement</h3><p>Observed "
        "inputs in one prespecified family are replaced by the outer training median while unavailable cells "
        "remain unavailable. These may create implausible combinations of correlated measurements. Changes "
        "describe this fitted model's input sensitivity, not a refitted ablation, a causal material effect or "
        "validated importance.</p><details><summary>Show all replacement probes</summary><div class=\"scroll\">" +
        table(replacements) + "</div></details>")
    image_html = _image_references(receipt, geometry_audit)
    css = """*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:#183447;background:#f3f6f8;margin:0}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{font-size:23px;margin-top:8px}h3{font-size:18px}.notice{background:#fff1d8;border-left:5px solid #d08225;padding:18px}section{background:#fff;padding:22px;margin:20px 0;border:1px solid #d9e4eb;border-radius:8px}img{display:block;max-width:100%;height:auto;margin:auto}figure{margin:20px 0}figcaption,.small{font-size:13px;color:#526b7a}.scroll{overflow:auto}.data{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:left;vertical-align:top}th{background:#edf3f6}a{color:#245db4}code{background:#edf3f6;padding:2px 4px}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.chart{margin:22px auto}nav a{display:inline-block;margin:4px 12px 4px 0}@media(max-width:700px){main{padding:15px}section{padding:15px}h1{font-size:27px}}"""
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E41S · Random Forest × geometry audit</title><style>{css}</style></head><body><main>
<h1>Random Forest × section geometry</h1><p>E41S · fixed two-by-two classifier experiment · 31 supplied crops, labels 7 / 7 / 17.</p>
<div class="notice"><strong>Crop diagnostics, with source dependence unresolved.</strong> The organisers constructed artificial visual groups from real crops taken from around 15 electrode sample images. Parent IDs are unavailable; related crops may appear on both sides of a fold. Counts and metrics below do not establish new-source accuracy, independent significance, actual supplier-lot identity or battery performance. Batch 3 remains the challenge reference.</div>
<nav><a href="#comparison">Model comparison</a><a href="#quality">Quality controls</a><a href="#sensitivity">Measurement sensitivity</a><a href="#failures">Failure explanations</a><a href="#geometry">Image references</a><a href="#provenance">Provenance</a></nav>
<section id="comparison"><h2>Model effect and feature effect</h2><p>{html.escape(headline)} All candidates were fixed before their evaluation; none is selected automatically by this page. Frozen submission v2 and QC remain unchanged.</p><p>The four main views share complete phase-quality gating, crop folds and fold-local median imputation. Existing 29 inputs versus existing 29 + eight geometry inputs is crossed with L1 logistic versus forest. The ungated L1 is a separate frozen-procedure reference. The forest is fixed at 500 trees, depth 3, leaf minimum 4, square-root feature sampling, balanced class weights, bootstrap sampling and seed 0. L1 scaling and regularisation selection remain inside training folds. No forest hyperparameter or seed search.</p><div class="scroll">{table(_display_summary(summary))}</div><img class="chart" src="{embed(figures[0])}" alt="Balanced accuracy and balanced log loss for all seven fixed candidates"><img class="chart" src="{embed(figures[1])}" alt="Four crop confusion matrices for the two-by-two model and geometry comparison"><img class="chart" src="{embed(figures[2])}" alt="All three class recalls for the four main candidates"><h3>Paired crop changes</h3><p>These compare the same withheld crops. They are descriptive; no new IID intervals or significance claims are reported.</p><div class="scroll">{table(paired)}</div><p>Scores are uncalibrated model outputs after weighted fitting, not probabilities of correctness. A forced bet is the largest score. Log loss and Brier show score quality as well as hard-label accuracy.</p></section>
<section id="quality"><h2>Quality and missingness controls</h2><p>Rejected mask-dependent inputs become unavailable before fold-local imputation. Bright, void and joint dependencies apply to the old and new inputs; values excluded by the gate cannot enter the fit. No crop ID, height, raw intensity level, quality flag or availability bit is a main-model predictor. Nonetheless imputation can encode quality patterns. The two dedicated forests test whether quality flags or the 37-cell availability pattern alone separate these labels; they are acquisition diagnostics and cannot be promoted as material evidence.</p><div class="scroll">{table(coverage)}</div><p>Observed-input counts describe measurement coverage and are never extra training samples. Imputed inputs are model assumptions, not observed geometry. The main crop count remains 31.</p><h3>Added geometry coverage</h3><div class="scroll">{table(geometry_coverage)}</div><p>Raw finite values can still be rejected by phase quality; these values are retained only for diagnostic inspection.</p></section>
<section id="sensitivity"><h2>Fixed measurement challenges</h2><p>The existing outer fits are used without refitting. Nominal measurements are compared with both mask thresholds shifted by −5 or +5 grayscale units and the bright-object floor raised from 50 to 100 px². The floor affects eligible shape and graph objects; it does not change raster-window bright fractions, void-width floor or bright–void correlation. These are probes of the added geometry panel, with the existing 29 descriptors held fixed. They are not end-to-end tests of acquisition changes, phase accuracy or the frozen QC pipeline.</p><div class="scroll">{table(_nuisance_summary(sensitivity))}</div><p>Binary phase masks are unvalidated. Sensitivity under a threshold perturbation can reveal a fragile measurement; stability cannot establish that the mask is correct.</p>{diagnostics_html}</section>
<section id="failures"><h2>Failures remain part of the evidence</h2><h3>Declared geometry-forest candidate: crop-held-out errors</h3><div class="scroll">{table(failure_table)}</div><h3>Revealed first-drop development diagnostics</h3><p>The three labels were already revealed before this experiment. These rows are development evidence and never an external test or a feature-selection objective. Every crop receives a bet, including low-contrast and unavailable-mask cases.</p><div class="scroll">{table(_feedback_table(feedback))}</div><h3>Local explanations of the declared candidate</h3><p>Forest path terms decompose the chosen-versus-comparison class score difference from the tree-root baseline. A positive term favours the chosen class. This exact accounting depends on tree path order and correlated inputs; it is not SHAP, a causal material effect or expert-validated feature importance. Terms from imputed cells remain marked unobserved. Linear models use decision-margin units, so their magnitudes cannot be compared directly with forest score units. Complete terms, baselines and reconstruction checks are saved in the explanation CSV.</p><details><summary>Show root baselines and score margins</summary><div class="scroll">{table(margins)}</div></details><details><summary>Show per-crop leading forest terms</summary><div class="scroll">{table(chosen_explanations[explanation_cols])}</div></details><p><a href="output/evaluation/explanations.csv">All explanation terms</a> · <a href="output/evaluation/known_oof_predictions.csv">All 31-crop out-of-fold scores</a> · <a href="output/evaluation/feedback_development_predictions.csv">All three development bets</a></p></section>
<section id="geometry"><h2>Eight measured image-geometry descriptors</h2><p>These reuse fixed section-geometry definitions from the morphology, void and neighbourhood audits. They remain experimental, threshold-dependent image measurements. Lengths are pixels. Bright fragments are not independently confirmed Si/SiOx particles, graph edges are centroid neighbourhoods rather than electrical contacts, and image x/y is not a confirmed through-thickness/collector orientation. No conductivity, tortuosity, capacity or battery-life value is inferred.</p><div class="scroll">{table(definitions)}</div>{image_html}</section>
<section id="provenance"><h2>Provenance and decision boundary</h2><p>All fitted candidates use only the original 31 known labels. Known results and the known-stage receipt are saved before the three first-drop feature rows are read. No new 34-crop submission is fitted, and no declared model or KPI trust status changes automatically. Source-image IDs would permit grouped outer and inner validation; the previous overlap audit recovered no groups and cannot detect non-overlapping crops from a shared parent.</p><p><a href="protocol.md">Fixed protocol</a> · <a href="output/geometry_known_input_receipt.json">Geometry extraction inputs</a> · <a href="output/geometry_known_verification.json">Geometry replay verification</a> · <a href="output/geometry_examples.json">Image-reference provenance</a> · <a href="output/evaluation/known_stage_receipt.json">Known-stage receipt</a> · <a href="output/evaluation/receipt.json">Complete model receipt</a> · <a href="output/evaluation/geometry_oof_sensitivity.csv">All measurement-challenge scores</a> · <a href="../source_crop_audit/findings.md">Source-overlap audit</a></p><p class="small">Scientific chart values and actual-image references are checked independently. Native browser layout remains unverified because browser policy previously blocked local-file navigation; no workaround was used.</p></section>
</main></body></html>'''
    (HERE / "report.html").write_text(page)
    print("Wrote report.html and three scientific charts from the saved E41S diagnostics")


if __name__ == "__main__":
    main()
