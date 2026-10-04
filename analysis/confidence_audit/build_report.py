"""Render E42S confidence diagnostics without calibrating or selecting a model."""
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
ORDER = ["legacy_l1", "base_l1", "gated_l2", "gated_lda", "appearance_l2", "augmented_l2",
         "base_rf", "geometry_l1", "geometry_rf", "geometry_l2"]
SHORT = {"legacy_l1": "Ungated L1", "base_l1": "Gated L1", "gated_l2": "Gated L2",
         "gated_lda": "Gated LDA", "appearance_l2": "Appearance L2", "augmented_l2": "Augmented L2",
         "base_rf": "Base forest", "geometry_l1": "Geometry L1", "geometry_rf": "Geometry forest",
         "geometry_l2": "Geometry L2"}
LABELS = {"legacy_l1": "Ungated L1 · preserved reference", "base_l1": "Gated L1 · 29 inputs",
          "gated_l2": "Gated L2 · 29 inputs", "gated_lda": "Gated LDA · 29 inputs",
          "appearance_l2": "Appearance L2 · 24 inputs", "augmented_l2": "Gated + appearance L2 · 53 inputs",
          "base_rf": "Gated forest · 29 inputs", "geometry_l1": "Gated geometry L1 · 37 inputs",
          "geometry_rf": "Gated geometry forest · 37 inputs", "geometry_l2": "New gated geometry L2 · 37 inputs"}


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data", escape=True,
                         float_format=lambda value: f"{value:.3f}")


def embed(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def read_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name)
    if frame.empty:
        raise ValueError(f"Missing report evidence: {name}")
    return frame


def validate_predictions(predictions: pd.DataFrame) -> None:
    """Check crop/score/bet identity before rendering a confidence claim."""
    if set(predictions.candidate) != set(ORDER):
        raise ValueError("Confidence report requires exactly the ten fixed measurement candidates")
    shared = None
    for name in ORDER:
        rows = predictions.loc[predictions.candidate.eq(name)]
        if len(rows) != 31 or rows.site.duplicated().any():
            raise ValueError(f"Expected 31 distinct withheld crops: {name}")
        if rows.true_batch.value_counts().to_dict() != {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}:
            raise ValueError(f"Incorrect visual-label composition: {name}")
        identities = set(zip(rows.site, rows.true_batch))
        if shared is not None and shared != identities:
            raise ValueError("Candidate confidence diagnostics use different crops/labels")
        shared = identities
        scores = rows[["score_" + c for c in CLASSES]].to_numpy(float)
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1):
            raise ValueError(f"Malformed class-score vectors: {name}")
        bets = np.asarray(CLASSES)[scores.argmax(axis=1)]
        if not np.array_equal(bets, rows.predicted_batch.to_numpy()):
            raise ValueError(f"Stored bet does not match the largest score: {name}")
        if not np.array_equal(bets == rows.true_batch.to_numpy(), rows.correct.to_numpy()):
            raise ValueError(f"Stored correctness target disagrees with labels: {name}")


def score_plot(summary: pd.DataFrame) -> Path:
    rows = summary.set_index("candidate").loc[ORDER]
    pairs = [("multiclass_logloss", "balanced_multiclass_logloss", "Full vector · multiclass log loss"),
             ("multiclass_brier", "balanced_multiclass_brier", "Full vector · multiclass Brier"),
             ("bet_nll", "balanced_bet_nll", "Top bet · correctness binary NLL"),
             ("bet_brier", "balanced_bet_brier", "Top bet · correctness binary Brier")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    y = np.arange(len(ORDER))
    for ax, (ordinary, balanced, title) in zip(axes.flat, pairs):
        ax.scatter(rows[ordinary], y - .12, color="#3c607a", marker="o", label="Observed-crop weighting", s=30)
        ax.scatter(rows[balanced], y + .12, color="#c0842d", marker="^", label="Equal-true-class weighting", s=35)
        ax.set_yticks(y, [SHORT[name] for name in ORDER], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Loss · lower is better")
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Confidence diagnostics · full class vector and own-bet correctness have different targets", fontsize=12)
    fig.tight_layout()
    path = OUT / "charts_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def outcome_score_plot(summary: pd.DataFrame) -> Path:
    rows = summary.set_index("candidate").loc[ORDER]
    y = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    ax.scatter(rows.mean_wrong_bet_score, y - .11, color="#a5486f", marker="x", s=55, label="Wrong bets · crop mean")
    ax.scatter(rows.mean_correct_bet_score, y + .11, color="#13885e", marker="o", s=35, label="Correct bets · crop mean")
    for i, name in enumerate(ORDER):
        ax.text(.995, i, f"wrong {int(rows.loc[name, 'n_wrong'])}; right {int(rows.loc[name, 'n_correct'])}",
                ha="right", va="center", fontsize=8, color="#526b79")
    ax.axvline(1 / 3, ls="--", color="#84929c", lw=1)
    ax.set_yticks(y, [SHORT[name] for name in ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(.31, 1.01)
    ax.set_xlabel("Mean largest model score · uncalibrated")
    ax.set_title("Weaker wrong bets can come with weaker right bets", loc="left", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    path = OUT / "charts_wrong_right.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def reliability_plot(bins: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ORDER):
        rows = bins.loc[bins.candidate.eq(name) & bins.n_sites.gt(0)]
        ax.plot([1 / 3, 1], [1 / 3, 1], color="#8c99a2", ls="--", lw=1)
        ax.scatter(rows.mean_bet_score, rows.empirical_accuracy, s=25 + 4 * rows.n_sites,
                   color="#315e83", alpha=.8, edgecolors="white", linewidth=.5)
        for _, row in rows.iterrows():
            ax.annotate(f"n={int(row.n_sites)}", (row.mean_bet_score, row.empirical_accuracy),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.set_title(SHORT[name], fontsize=10, loc="left")
        ax.set_xlim(.30, 1.02)
        ax.set_ylim(-.04, 1.09)
        ax.set_xticks([1 / 3, .5, .7, .9], [".33", ".5", ".7", ".9"])
        ax.grid(alpha=.18)
    for ax in axes[1]:
        ax.set_xlabel("Mean largest score", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed fraction correct", fontsize=9)
    fig.suptitle("Fixed-bin observed-crop reliability · points are occupied bins; n and class mix limit interpretation", fontsize=13)
    fig.tight_layout()
    path = OUT / "charts_reliability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def renamed(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    if "candidate" in result:
        result.candidate = result.candidate.map(lambda name: LABELS.get(name, name))
    return result


def main() -> None:
    summary = read_csv("confidence_summary.csv")
    predictions = read_csv("predictions.csv")
    known = predictions.loc[predictions.cohort.eq("known_oof")]
    validate_predictions(known)
    known_summary = summary.loc[summary.cohort.eq("known_oof")]
    if known_summary.candidate.duplicated().any() or set(known_summary.candidate) != set(ORDER):
        raise ValueError("Confidence summaries do not have the fixed ten known candidates")
    known_summary = known_summary.set_index("candidate").loc[ORDER].reset_index()
    reliability = read_csv("reliability_bins.csv")
    known_bins = reliability.loc[reliability.cohort.eq("known_oof")]
    for name in ORDER:
        bins = known_bins.loc[known_bins.candidate.eq(name)]
        if len(bins) != 5 or int(bins.n_sites.sum()) != 31:
            raise ValueError(f"Fixed-bin occupancy is inconsistent: {name}")
        if not np.array_equal(bins[["n_Batch_1", "n_Batch_2", "n_Batch_3"]].sum(axis=1), bins.n_sites):
            raise ValueError(f"Fixed-bin class composition is inconsistent: {name}")
    thresholds = read_csv("threshold_counts.csv")
    paired = read_csv("paired_changes.csv")
    controls = read_csv("controls_summary.csv")
    explanations = read_csv("explanations.csv")
    probe_summary = read_csv("probe_known_summary.csv").iloc[0]
    for name in ORDER:
        row = known_summary.loc[known_summary.candidate.eq(name)].iloc[0]
        actual = known.loc[known.candidate.eq(name)]
        if int(row.n_correct) != int(actual.correct.sum()) or int(row.n_sites) != len(actual):
            raise ValueError(f"Confidence summary count mismatch: {name}")
    figures = [score_plot(known_summary), outcome_score_plot(known_summary), reliability_plot(known_bins)]
    hard_columns = ["candidate", "n_correct", "n_sites", "accuracy", "balanced_accuracy", "n_wrong", "mean_bet_score"]
    full_columns = ["candidate", "multiclass_logloss", "balanced_multiclass_logloss", "multiclass_brier", "balanced_multiclass_brier", "mean_true_class_score"]
    bet_columns = ["candidate", "bet_brier", "balanced_bet_brier", "bet_nll", "balanced_bet_nll", "mean_wrong_bet_score", "mean_correct_bet_score"]
    feedback = predictions.loc[predictions.cohort.eq("feedback_development")].copy()
    feedback = feedback.sort_values(["candidate", "site"], key=lambda s: s.map({name: i for i, name in enumerate(ORDER)}) if s.name == "candidate" else s)
    feedback_columns = ["candidate", "site", "true_batch", "predicted_batch", "correct", "bet_score", "true_class_score", "n_observed_inputs"]
    baseline_drop = feedback.loc[feedback.candidate.eq("legacy_l1")].set_index("site")
    l2_drop = feedback.loc[feedback.candidate.eq("gated_l2")].set_index("site")
    if set(baseline_drop.index) != set(l2_drop.index) or len(baseline_drop) != 3:
        raise ValueError("Missing three shared development rows for same-bet score comparison")
    development_comparison = pd.DataFrame([{"Crop": site,
        "Truth": row.true_batch, "Reference bet": row.predicted_batch,
        "Gated L2 bet": l2_drop.loc[site].predicted_batch,
        "Reference correct": bool(row.correct), "Gated L2 correct": bool(l2_drop.loc[site].correct),
        "Reference bet score": row.bet_score, "Gated L2 bet score": l2_drop.loc[site].bet_score,
        "Bet-score change": l2_drop.loc[site].bet_score - row.bet_score}
        for site, row in baseline_drop.iterrows()])
    if not np.array_equal(development_comparison["Reference bet"], development_comparison["Gated L2 bet"]):
        raise ValueError("The declared development comparison no longer has unchanged bets")
    margin_cols = ["cohort", "site", "predicted_batch", "comparison_batch", "root_or_intercept_margin", "total_margin"]
    margins = explanations[margin_cols].drop_duplicates()
    terms = explanations.loc[explanations["rank"].le(5)]
    term_cols = ["cohort", "site", "predicted_batch", "comparison_batch", "feature", "family", "feature_group", "contribution", "observed"]
    primary_thresholds = thresholds.loc[thresholds.candidate.eq("geometry_l2") & thresholds.cohort.eq("known_oof")]
    known_paired = paired.loc[paired.cohort.eq("known_oof")]
    new_vs_l2 = known_paired.loc[known_paired.candidate.eq("geometry_l2") & known_paired.reference.eq("gated_l2")]
    vector_control_columns = ["control", "cohort", "n_sites", "multiclass_logloss", "balanced_multiclass_logloss",
                              "multiclass_brier", "balanced_multiclass_brier"]
    vector_controls = controls.loc[controls.control.isin(["uniform_vector", "train_prior_vector"]), vector_control_columns]
    flatten_control_columns = ["candidate", "cohort", "n_correct", "n_wrong", "mean_bet_score",
                              "bet_brier", "balanced_bet_brier", "bet_nll", "balanced_bet_nll"]
    confidence_controls = controls.loc[controls.control.eq("uniform_confidence_same_bets"), flatten_control_columns]
    prior_bet_columns = ["cohort", "n_sites", "n_correct", "accuracy", "balanced_accuracy", "mean_bet_score",
                        "bet_brier", "balanced_bet_brier", "bet_nll", "balanced_bet_nll"]
    prior_bets = controls.loc[controls.control.eq("train_prior_vector"), prior_bet_columns]
    comparison_sentence = ""
    if len(new_vs_l2) == 1:
        row = new_vs_l2.iloc[0]
        comparison_sentence = (f"Against gated L2 with 29 inputs, geometry L2 changes {int(row.changed_bets)} crop bets, "
            f"with {int(row.newly_correct)} newly correct and {int(row.newly_wrong)} newly wrong. "
            f"Equal-true-class multiclass log loss changes by {row.balanced_multiclass_logloss_delta:+.3f}.")
    css = """*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:#183447;background:#f3f6f8;margin:0}main{max-width:1240px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{font-size:23px;margin:8px 0 14px}h3{font-size:18px}.notice{background:#fff1d8;border-left:5px solid #d08225;padding:18px}section{background:white;padding:22px;margin:20px 0;border:1px solid #d9e4eb;border-radius:8px}img{display:block;max-width:100%;height:auto;margin:22px auto}.scroll{overflow:auto}.data{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:left;vertical-align:top}th{background:#edf3f6}a{color:#245db4}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.small{font-size:13px;color:#526b7a}nav a{display:inline-block;margin:4px 12px 4px 0}@media(max-width:700px){main{padding:15px}section{padding:15px}h1{font-size:27px}}"""
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E42S · Bet, confidence and explanation audit</title><style>{css}</style></head><body><main>
<h1>Bet, confidence and explanation</h1><p>E42S / D64S · one additional fixed geometry L2 candidate and ten deduplicated measurement-model diagnostics.</p>
<div class="notice"><strong>Model scores are candidate confidence, not calibrated correctness probabilities.</strong> Judging includes the batch bet, confidence quality and explanations. Exact confidence semantics, high/low cutoffs and the complete payoff rule are unavailable. This page measures predictive score behaviour without inventing an organiser utility function or changing the submission.</div>
<nav><a href="#probe">Geometry L2 probe</a><a href="#scores">Score quality</a><a href="#reliability">Reliability</a><a href="#flattening">Flattening controls</a><a href="#failures">Development examples</a><a href="#provenance">Provenance</a></nav>
<section id="probe"><h2>One new, fixed L2 geometry fit</h2><p>The new model uses the same complete phase-quality gate, existing 29 inputs and exact eight section-geometry descriptors from E41S, with multinomial L2 logistic regression. Training-fold median imputation, scaling and C selection remain inside each crop holdout. No feature, geometry, seed, forest-grid or confidence-weight search. Known-only models and evidence are saved before loading the three already-revealed development examples.</p><p>New geometry L2: balanced accuracy {probe_summary.balanced_accuracy:.3f}; class recall {probe_summary.recall_Batch_1:.3f} / {probe_summary.recall_Batch_2:.3f} / {probe_summary.recall_Batch_3:.3f}; {int(probe_summary.n_correct)} / {int(probe_summary.n_sites)} correct crop bets. {html.escape(comparison_sentence)}</p><div class="scroll">{table(renamed(known_summary, hard_columns))}</div><p>Acquisition-only and availability-only forests are retained in E41S as controls and excluded from these ten measurement candidates. Identical legacy/base L1 outputs reused across experiments appear once.</p></section>
<section id="scores"><h2>Full-vector quality and own-bet confidence answer different questions</h2><p>Multiclass log loss and Brier compare all three class scores with the supplied true label. The binary correctness diagnostics use q = largest model score and z = whether that candidate's forced bet was correct. Bets can change between models, so z can also change: binary score comparisons do not share a fixed correctness target and cannot alone rank the material classifier or challenge utility. Every comparison keeps hard-label outcomes and full-vector metrics visible.</p><p>Observed-crop weighting gives each of 31 crops equal weight. Equal-true-class weighting gives each supplied class equal total weight despite 7 / 7 / 17 crop counts. All four losses are lower-is-better descriptive diagnostics. No correctness calibrator is fitted.</p><img src="{embed(figures[0])}" alt="Ten candidates compared on multiclass and binary correctness losses under two weighting schemes"><h3>Full class vector</h3><div class="scroll">{table(renamed(known_summary, full_columns))}</div><h3>Confidence in each candidate's own bet</h3><div class="scroll">{table(renamed(known_summary, bet_columns))}</div><img src="{embed(figures[1])}" alt="Mean model score on wrong and correct crop bets for each candidate"><p>The wrong/right mean scores above are simple crop averages. Lower wrong-bet scores can help error honesty while lower right-bet scores weaken successful assignments. Neither movement alone establishes better confidence, calibration or organiser reward.</p><details><summary>Paired crop changes against fixed references</summary><p>Same-wrong and same-correct subsets keep the correctness outcome unchanged when comparing bet-score shifts. Changed bets, true-class scores and full-vector losses remain separate.</p><div class="scroll">{table(known_paired)}</div></details></section>
<section id="reliability"><h2>Fixed-bin reliability, with occupancy visible</h2><p>The bins were fixed at 1/3, 0.5, 0.6, 0.7, 0.8 and 1. Points show the observed crop fraction correct versus mean largest score for occupied bins; the diagonal describes agreement between those quantities. Marker size and labels show occupancy. Empty bins provide no evidence; class composition is saved for every bin. These sparse, dependent summaries do not validate a percentage-correct interpretation.</p><img src="{embed(figures[2])}" alt="Fixed-bin reliability diagrams for all ten candidates with crop occupancy labels"><details><summary>All occupied and empty bins with class composition</summary><div class="scroll">{table(known_bins)}</div></details><h3>Fixed threshold diagnostics for geometry L2</h3><p>Thresholds 0.5 / 0.6 / 0.7 / 0.8 / 0.9 count high-score right and wrong bets and coverage. They are sensitivity views, not a selected cutoff, release threshold or assumed judging payoff.</p><div class="scroll">{table(primary_thresholds)}</div><details><summary>All candidate and development threshold counts</summary><div class="scroll">{table(thresholds)}</div></details></section>
<section id="flattening"><h2>Lower scores are not sufficient evidence of improvement</h2><p>The uniform three-class vector is a full-vector flattening control. Its separate confidence control sets q to 1/3 while retaining each candidate's original bet and correctness target; it is explicitly not a new argmax classifier. The outer-training class-prior vector is a separate baseline, estimated only from the training fold. Neither uses a withheld label to estimate its prior.</p><h3>Full-vector controls</h3><div class="scroll">{table(vector_controls)}</div><p>The uniform-vector argmax tie is incidental and has no classification interpretation here; only its proper full-vector scores are displayed.</p><h3>Outer-training-prior forced bets</h3><div class="scroll">{table(prior_bets)}</div><h3>Flattened confidence with each original bet preserved</h3><div class="scroll">{table(renamed(confidence_controls, flatten_control_columns))}</div><p>These controls expose the cost of globally weakening scores. Missing confidence definitions, cutoffs and payouts—especially the low-confidence-correct outcome—prevent computing the organiser's expected utility from the illustrative judging examples.</p></section>
<section id="failures"><h2>Revealed development examples and explanations</h2><p>The three first-drop labels were known before this work. All candidate bets and true-class scores remain development evidence; no model, confidence threshold or calibration transform is selected from them.</p><h3>A favourable confidence pattern on the same three development bets</h3><div class="scroll">{table(development_comparison)}</div><p>Gated L2 keeps these three reference bets, reduces both wrong-bet scores and increases the correct-bet score. This is a useful development observation, not a calibration or reward guarantee. The known-crop table above still favours the legacy model, and the three revealed examples cannot select a model.</p><details><summary>All ten candidates on the three revealed examples</summary><div class="scroll">{table(renamed(feedback, feedback_columns))}</div></details><h3>Geometry L2: exact linear explanations</h3><p>Terms plus intercept reconstruct the chosen-versus-comparison decision margin. They use margin units rather than class-score units, are noncausal and do not validate the phase measurements. Imputed inputs are marked unobserved. Explanations account for what this model uses; they are not a measured judge-quality score. The display derivative corrects an inherited family label for the eight mask-dependent geometry inputs while retaining the original as-run label. Feature groups, measured inputs, terms and reconstructed margins are unchanged; the as-run CSV remains preserved.</p><details><summary>Root/intercept baselines and complete margins</summary><div class="scroll">{table(margins)}</div></details><details><summary>Leading observed and imputed terms per crop</summary><div class="scroll">{table(terms[term_cols])}</div></details><p><a href="output/explanations.csv">All geometry L2 terms with corrected display metadata</a> · <a href="output/probe_explanations.csv">Preserved as-run terms</a> · <a href="output/explanation_metadata_correction.json">Metadata correction receipt</a> · <a href="../forest_geometry/report.html#geometry">Actual geometry overlays, values and phase-quality eligibility</a></p></section>
<section id="provenance"><h2>What this evidence permits</h2><p>The real electrode crops were organised into artificial visual batches from around 15 source images. Parent IDs remain unknown; related crops may cross folds. Every score and reliability view is a crop-held-out development diagnostic, not new-source accuracy, independent significance, actual supplier-lot identity or battery performance. No new IID interval or p-value, no expert mask accuracy, no 34-crop fit and no automatic model or trust promotion.</p><p>Current submission and QC remain unchanged. A future model or correctness-confidence revision needs a separately recorded version and appropriate new-source evidence. A change to score sharpness must be assessed on wrong and right bets together.</p><p><a href="protocol.md">Fixed protocol</a> · <a href="output/known_stage_receipt.json">Known-only probe receipt</a> · <a href="output/probe_receipt.json">Probe provenance</a> · <a href="output/audit_receipt.json">Confidence audit provenance</a> · <a href="output/predictions.csv">All candidate predictions</a> · <a href="output/confidence_summary.csv">All confidence metrics</a> · <a href="output/paired_changes.csv">All paired changes</a></p><p class="small">Scientific chart values and saved evidence are checked independently. Native browser layout remains unverified because local-file navigation was previously blocked; no workaround was used.</p></section>
</main></body></html>'''
    (HERE / "report.html").write_text(page)
    print("Wrote E42S report.html and three confidence charts; no calibration or model selection")


if __name__ == "__main__":
    main()
