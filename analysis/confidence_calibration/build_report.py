"""Render E43S scalar-score evidence; no fitting, threshold or model selection."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
ORDER = ["original", "calibrated", "fixed_softening", "uniform_control"]
LABELS = {"original": "Original frozen v2", "calibrated": "Fitted scalar · nested evaluation",
          "fixed_softening": "Fixed α = 0.5 · control", "uniform_control": "Uniform scores · original bets retained"}
SCORE_COLUMNS = ["score_" + cls for cls in CLASSES]


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data", escape=True,
                         float_format=lambda value: f"{value:.3f}")


def embed(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def read_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name)
    if frame.empty:
        raise ValueError(f"Report evidence is empty: {name}")
    return frame


def validate_predictions(frame: pd.DataFrame, n_sites: int) -> pd.DataFrame:
    if set(frame.candidate) != set(ORDER):
        raise ValueError("Scalar comparison requires the four declared candidates/controls")
    base = frame.loc[frame.candidate.eq("original")].set_index("site")
    if len(base) != n_sites or base.index.duplicated().any():
        raise ValueError("Expected one original score row for every crop")
    for name in ORDER:
        rows = frame.loc[frame.candidate.eq(name)].set_index("site")
        if len(rows) != n_sites or rows.index.duplicated().any() or set(rows.index) != set(base.index):
            raise ValueError(f"Candidate crop coverage disagrees: {name}")
        rows = rows.reindex(base.index)
        if not rows.true_batch.equals(base.true_batch) or not rows.predicted_batch.equals(base.predicted_batch):
            raise ValueError(f"Scalar changed crop labels or forced bets: {name}")
        scores = rows[SCORE_COLUMNS].to_numpy(float)
        if not np.isfinite(scores).all() or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1):
            raise ValueError(f"Invalid normalized class scores: {name}")
        # The uniform control explicitly retains the original bet outside its tied vector.
        if name != "uniform_control" and not np.array_equal(np.asarray(CLASSES)[scores.argmax(axis=1)], rows.predicted_batch):
            raise ValueError(f"Positive scalar or fixed-softening argmax changed: {name}")
    if n_sites == 31 and base.true_batch.value_counts().to_dict() != {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}:
        raise ValueError("Known crop label composition changed")
    return base


def paired_score_plot(known: pd.DataFrame) -> Path:
    base = known.loc[known.candidate.eq("original")].set_index("site")
    corrected = known.loc[known.candidate.eq("calibrated")].set_index("site").reindex(base.index)
    x = base[SCORE_COLUMNS].max(axis=1).to_numpy(float)
    y = corrected[SCORE_COLUMNS].max(axis=1).to_numpy(float)
    correct = base.true_batch.eq(base.predicted_batch).to_numpy(bool)
    fig, ax = plt.subplots(figsize=(7.4, 6))
    ax.plot([1 / 3, 1], [1 / 3, 1], color="#8999a3", ls="--", lw=1, label="Score unchanged")
    ax.scatter(x[correct], y[correct], marker="o", s=45, color="#16845d",
               edgecolors="white", linewidth=.5, label=f"Correct bet · {int(correct.sum())} crops")
    ax.scatter(x[~correct], y[~correct], marker="x", s=65, color="#a44569",
               linewidth=1.5, label=f"Wrong bet · {int((~correct).sum())} crops")
    ax.set_xlim(.31, 1.02); ax.set_ylim(.31, 1.02)
    ax.set_xlabel("Original largest normalized OvR score")
    ax.set_ylabel("Largest score after nested scalar correction")
    ax.set_title("Same bets; held-out crop scores can sharpen or soften", loc="left", fontsize=12)
    ax.grid(alpha=.18); ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    path = OUT / "plots_paired_scores.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def losses_plot(summary: pd.DataFrame) -> Path:
    rows = summary.set_index("candidate").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    y = np.arange(len(ORDER))
    pairs = [("multiclass_logloss", "balanced_multiclass_logloss", "Full-vector multiclass log loss"),
             ("bet_brier", "balanced_bet_brier", "Own-bet correctness Brier")]
    for ax, (ordinary, balanced, title) in zip(axes, pairs):
        ax.scatter(rows[ordinary], y - .12, s=40, color="#3d617b", marker="o", label="Observed-crop weighting")
        ax.scatter(rows[balanced], y + .12, s=45, color="#c0822c", marker="^", label="Equal-true-class weighting")
        ax.set_yticks(y, ["Original v2", "Nested scalar", "Fixed α = .5", "Uniform control"], fontsize=9)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Loss · lower is better")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("31 crop diagnostics · identical bets give identical binary correctness targets", fontsize=12)
    fig.tight_layout()
    path = OUT / "plots_losses.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def score_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["correct"] = rows.true_batch.eq(rows.predicted_batch)
    rows["bet_score"] = [row["score_" + row.predicted_batch] for _, row in rows.iterrows()]
    rows["true_class_score"] = [row["score_" + row.true_batch] for _, row in rows.iterrows()]
    rows.candidate = rows.candidate.map(LABELS)
    return rows[["site", "candidate", "true_batch", "predicted_batch", "correct", "alpha", "bet_score", "true_class_score"]]


def decision_text(promotion: dict) -> str:
    if promotion["promoted"]:
        return "Declared outcome: the fitted scalar is promoted through a separately recorded model version, preserving v2."
    if promotion["eligible_for_versioned_replacement"]:
        return "Declared outcome: the criterion is met and the scalar is eligible; its versioned replacement has not been completed."
    return "Declared outcome: the criterion is not met; the original frozen v2 remains the submission."


def main() -> None:
    known = read_csv("known_predictions.csv")
    feedback = read_csv("feedback_predictions.csv")
    validate_predictions(known, 31); validate_predictions(feedback, 3)
    summary = read_csv("summary.csv")
    known_summary = summary.loc[summary.cohort.eq("known_oof")].set_index("candidate").loc[ORDER].reset_index()
    temperatures = read_csv("temperatures.csv")
    full_rows = temperatures.loc[temperatures.fold.astype(str).eq("full")]
    if len(temperatures) != 32 or len(full_rows) != 1:
        raise ValueError("Expected 31 outer scalars plus one deployment-only scalar")
    full = full_rows.iloc[0]
    if not np.isfinite(temperatures.alpha).all() or not temperatures.alpha.between(.05, 20).all():
        raise ValueError("Fitted inverse temperatures are outside the fixed bounds")
    direction = "sharpens" if full.alpha > 1 else "softens" if full.alpha < 1 else "preserves"
    promotion = json.loads((OUT / "promotion.json").read_text())
    historical = json.loads((OUT / "historical_comparator_difference.json").read_text())
    if not promotion["all_bets_unchanged"]:
        raise ValueError("Promotion record reports changed bets in an argmax-preserving experiment")
    loss_names = ["multiclass_logloss", "balanced_multiclass_logloss", "multiclass_brier", "balanced_multiclass_brier",
                  "bet_brier", "balanced_bet_brier", "bet_nll", "balanced_bet_nll"]
    metrics = known_summary[["candidate", "n_correct", "n_sites", "balanced_accuracy", *loss_names]].copy()
    metrics.candidate = metrics.candidate.map(LABELS)
    reductions = pd.DataFrame([{"Gate metric": name, "Relative reduction": value,
        "Required": promotion.get("minimum_relative_reduction", .01), "Pass": bool(value >= promotion.get("minimum_relative_reduction", .01))}
        for name, value in promotion["relative_primary_reductions"].items()])
    other_losses = pd.DataFrame([{"Other loss": name, "Nonworsening within tolerance": bool(passed)}
        for name, passed in promotion["other_loss_nonworsening"].items()])
    figures = [paired_score_plot(known), losses_plot(known_summary)]
    outcome = decision_text(promotion)
    original_metrics = known_summary.loc[known_summary.candidate.eq("original")].iloc[0]
    fitted_metrics = known_summary.loc[known_summary.candidate.eq("calibrated")].iloc[0]
    tradeoff = (f"Equal-class multiclass log loss changes {original_metrics.balanced_multiclass_logloss:.3f} → "
        f"{fitted_metrics.balanced_multiclass_logloss:.3f}, while equal-class own-bet Brier changes "
        f"{original_metrics.balanced_bet_brier:.3f} → {fitted_metrics.balanced_bet_brier:.3f}. "
        f"Mean wrong-bet score falls {original_metrics.mean_wrong_bet_score:.3f} → {fitted_metrics.mean_wrong_bet_score:.3f}; "
        f"mean right-bet score also falls {original_metrics.mean_correct_bet_score:.3f} → {fitted_metrics.mean_correct_bet_score:.3f}.")
    css = """*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:#183447;background:#f3f6f8;margin:0}main{max-width:1180px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{font-size:23px;margin:8px 0 14px}h3{font-size:18px}.notice{background:#fff1d8;border-left:5px solid #d08225;padding:18px}section{background:#fff;padding:22px;margin:20px 0;border:1px solid #d9e4eb;border-radius:8px}img{display:block;max-width:100%;height:auto;margin:22px auto}.scroll{overflow:auto}.data{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:left;vertical-align:top}th{background:#edf3f6}a{color:#245db4}code{background:#edf3f6;padding:2px 4px}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.small{font-size:13px;color:#526b7a}nav a{display:inline-block;margin:4px 12px 4px 0}@media(max-width:700px){main{padding:15px}section{padding:15px}h1{font-size:27px}}"""
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E43S · Scalar confidence-layer audit</title><style>{css}</style></head><body><main>
<h1>One scalar confidence layer for frozen v2</h1><p>E43S / D67S · same material classifier and forced batch bets · score correction evaluated inside crop holdouts.</p>
<div class="notice"><strong>{html.escape(outcome)}</strong><p>Unknown parent-image IDs remain a limit on all results. Related crops from around 15 source images may cross folds. A crop-level score improvement does not establish confidence as a probability of correctness on new sources.</p></div>
<nav><a href="#method">Method</a><a href="#results">Known-crop results</a><a href="#decision">Declared gate</a><a href="#development">Revealed examples</a><a href="#provenance">Provenance</a></nav>
<section id="method"><h2>Correct the normalized score vector, preserve the classifier</h2><p>The frozen v2 classifier produces normalized one-versus-rest probabilities from liblinear. The layer uses <code>q = softmax(α × log(p))</code>, with positive inverse temperature <code>α = 1 / T</code>. Identity α = 1 reproduces the original vector. It operates on normalized OvR log probabilities, not raw decision-function logits. A positive scalar preserves every class ranking and the forced bet. It can sharpen or soften; the fit determines the direction.</p><p>For each outer crop holdout, α is fitted only to internal leave-one-crop-out scores of the other 30 crops, from classifiers trained on 29 crops. The 465 paired-deletion classifier fits are reused symmetrically. Regularisation selection, imputation and scaling remain training-local. The omitted outer crop never enters its scalar fit. The initial three-fold draft was corrected before any fit; the final protocol records that change and the remaining one-crop training-size difference.</p><p>A separate deployment scalar is fitted to the 31 frozen outer-holdout vectors from 30-crop models. Its fit on those 31 is not used as held-out calibration evidence. Known results below use the 31 outer-specific scalars. All models and the decision receipt are saved before reading the three revealed development examples.</p></section>
<section id="results"><h2>Same bets, different scores</h2><p>Correctness targets remain identical across the original, fitted scalar and fixed-softening candidate. The uniform control retains the original bets outside its tied score vector; it is a flattening diagnostic and cannot be promoted. Scores remain experimental class outputs, not proven correctness confidence.</p><p><strong>{html.escape(tradeoff)}</strong> The deployment-only inverse temperature is α = {full.alpha:.3f} (T = {full.temperature:.3f}); it {direction} the scores. This is a tradeoff in predictive losses, not an established calibration improvement.</p><img src="{embed(figures[0])}" alt="Original versus nested scalar top-bet scores, with distinct symbols for correct and wrong crop bets"><img src="{embed(figures[1])}" alt="Original, fitted scalar and controls compared on multiclass log loss and own-bet Brier under two weighting schemes"><div class="scroll">{table(metrics)}</div><p>Multiclass losses evaluate the whole class vector; correctness Brier and binary NLL evaluate the largest score against the unchanged bet's correctness. Observed-crop weighting counts every crop equally; equal-true-class weighting gives Batches 1 / 2 / 3 equal total weight despite 7 / 7 / 17 crop counts. Lower is better for every reported loss.</p><details><summary>All fitted inverse temperatures</summary><p>Thirty-one outer scalars provide held-out diagnostics; the full scalar is deployment-only. Alpha bounds are 0.05 to 20, with identity and endpoints checked and numerical ties preferring identity.</p><div class="scroll">{table(temperatures)}</div></details></section>
<section id="decision"><h2>Declared practical promotion gate</h2><p>The rule was fixed before fitting: at least 1% relative reduction in both equal-class multiclass log loss and equal-class own-bet Brier, no increase beyond 10⁻⁶ in any other six loss measures, unchanged bets, finite scores and complete replay/provenance. Only the fitted scalar is eligible. Controls are not selected.</p><div class="scroll">{table(reductions)}</div><div class="scroll">{table(other_losses)}</div><p><strong>{html.escape(outcome)}</strong> This is a declared practical criterion, not a significance test, calibrated correctness guarantee or inferred organiser grading function. Exact challenge payoff and high/low-confidence definitions remain unknown.</p><p><a href="output/promotion.json">Complete decision record</a></p></section>
    <section id="development"><h2>Three already-revealed examples</h2><p>The same known-only full classifier and deployment scalar score these examples after the known-stage decision. Their labels are development diagnostics only; they do not fit α, set a criterion or select a replacement. A changed score leaves the original bet and any mistake intact.</p><div class="scroll">{table(score_table(feedback))}</div><p>The uniform control is defined directly; it is not a fitted positive-temperature layer.</p></section>
<section id="provenance"><h2>Inherited evidence and limits</h2><p>No feature, mask, quality policy, label override, acquisition statistic or confidence cutoff changes. The layer creates no new material evidence. Existing texture and phase-mask limitations remain. The dataset's artificial visual labels are not real supplier-lot or battery-performance truth; crop counts are not independent source/specimen counts. No new IID interval, p-value, human mask validation or 34-crop fit is claimed.</p><p><strong>Exact-baseline provenance:</strong> the previous E42S matched comparator is not bit-exact frozen v2 on crop <code>epqdaau9</code>. The saved classifier uses C = 2 there while the anchor-ordered comparator uses C = 0.5; maximum class-score difference is {historical["max_score_difference"]:.6f}, with no changed bets. Its historical log loss cannot be quoted as actual v2 evidence. This audit replays the saved v2 deletion models and its own sealed score table; earlier artifacts remain preserved. <a href="output/historical_comparator_difference.json">Recorded comparator difference</a>.</p><p>Use the frozen classifier's feature explanations for the unchanged batch bet. A scalar correction of the normalized output does not supply new feature contributions or validate the measurements.</p><p><a href="../submission_v2/README.md">Frozen v2 provenance and limitations</a> · <a href="../submission_v2/first_run/scored/driver_contrasts.csv">Preserved feature driver contrasts</a> · <a href="../submission_v2/first_run/scored/explanations.json">Preserved feature explanations</a> · <a href="protocol.md">Final fixed protocol</a> · <a href="protocol_initial_threefold.md">Retained initial draft</a> · <a href="output/known_predictions.csv">All known score vectors</a> · <a href="output/feedback_predictions.csv">All development vectors</a> · <a href="README.md">Run and evidence guide</a> · <a href="output/known_receipt.json">Known-before-feedback receipt</a> · <a href="output/receipt.json">Complete study receipt</a></p><p class="small">Scientific figures and saved score identities are checked separately. Native browser layout remains unverified because local-file navigation was previously blocked; no workaround was used.</p></section>
</main></body></html>'''
    (HERE / "report.html").write_text(page)
    readme = f'''# E43S: scalar score correction for frozen v2

{outcome}

The deployment-only inverse temperature is **α = {full.alpha:.6f}**
(T = {full.temperature:.6f}); it {direction} the normalized score vector.
The 31 known evaluation rows use distinct outer-fold scalars, not this full fit.
All forced batch bets and correctness targets are preserved.

{tradeoff} This tradeoff does not establish better correctness calibration.

This layer applies `softmax(alpha * log(p))` to **normalized OvR probabilities**
from the saved D52 v2 material classifier. Alpha 1 reproduces the original scores.
It does not apply softmax to raw classifier logits and creates no new material
features or feature attributions. Scores are a fitted correction; new-source
correctness calibration remains unproved.

## Evidence

- [HTML report](report.html): original/corrected paired scores, losses, promotion
  checks, fitted temperatures and all three revealed development examples.
- [Final protocol](protocol.md) and [initial draft](protocol_initial_threefold.md):
  the reviewer-driven change from three-fold internal scoring to paired internal
  crop-LOO was made before fitting.
- [Known scores](output/known_predictions.csv),
  [development scores](output/feedback_predictions.csv),
  [all summary losses](output/summary.csv) and [temperature fits](output/temperatures.csv).
- [Declared promotion gate](output/promotion.json),
  [known-before-feedback receipt](output/known_receipt.json) and
  [complete receipt](output/receipt.json).
- [Preserved v2 model and limitations](../submission_v2/README.md),
  [feature drivers](../submission_v2/first_run/scored/driver_contrasts.csv) and
  [feature explanations](../submission_v2/first_run/scored/explanations.json).

The practical gate requires at least 1% relative improvement in equal-class
multiclass log loss and own-bet correctness Brier, no worsening beyond 1e-6 in
the other six losses, unchanged bets and complete provenance/replay. Only the
fitted scalar can qualify; fixed softening and uniform scores are controls.
Exact organiser utility and confidence cutoffs are unknown.

The E42S matched comparator differs from saved frozen v2 on `epqdaau9`
(C = 0.5 versus C = 2), despite unchanged bets. Its historical log loss
cannot be reused as the actual-v2 baseline. This study replays the saved v2
deletions and sealed v2 score table; the
[comparator difference](output/historical_comparator_difference.json) is retained.

The artificial visual classes contain crops from around 15 source images.
Parent IDs remain unknown, so related crops can cross folds. Crop-held-out losses
are descriptive development evidence, not source-held-out correctness confidence
or manufacturing-performance validation. The three revealed labels were read
after the known-stage decision and never fit or select alpha.

## Render saved evidence

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/polaron-calibration-report-mpl /opt/anaconda3/bin/python3 -m analysis.confidence_calibration.build_report
```

This command reads the saved results and writes the report, two scientific plots
and this guide. It performs no classifier or scalar fit. The experiment producer
`analysis.confidence_calibration.run` refuses an existing output directory;
preserve the as-run results rather than overwriting them.

Native browser layout remains unverified because local-file navigation was
previously blocked. Scientific charts, saved score identities and links are
checked separately; no browser workaround is used.
'''
    (HERE / "README.md").write_text(readme)
    print("Wrote E43S report.html and two scientific plots; no classifier or calibration fit performed")


if __name__ == "__main__":
    main()
