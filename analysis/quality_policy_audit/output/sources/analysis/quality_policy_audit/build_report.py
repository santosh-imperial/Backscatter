"""Build the E37S dependency/coverage report and unreviewed boundary examples."""
from __future__ import annotations

import base64
from html import escape
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from analysis.submission_v2.runtime import load_runtime
from . import policy as qp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
COLORS = {"Batch_1":"#2a78d6", "Batch_2":"#eb6834", "Batch_3":"#1baf7a"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_uri(path):
    kind = "jpeg" if path.suffix == ".jpg" else "png"
    return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    receipt = json.loads((OUT / "receipt.json").read_text())
    for rel, expected in receipt["artifact_sha256"].items():
        assert sha(OUT / rel) == expected, rel
    assets = HERE / "assets"; assets.mkdir(exist_ok=True)
    known = pd.read_csv(OUT / "known_sites.csv").sort_values(["batch", "site"])
    summary = pd.read_csv(OUT / "known_summary.csv")
    failure = pd.read_csv(OUT / "failure_examples.csv")
    eligibility = pd.read_csv(OUT / "known_input_eligibility.csv").query("variant == 'dependency_gate'")
    dependencies = pd.read_csv(OUT / "feature_dependencies.csv")
    delta = json.loads((OUT / "paired_comparison.json").read_text())
    cells = eligibility.pivot(index="feature", columns="site", values="observed_model_input").loc[qp.FEATURES, known.site]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(cells.to_numpy(int), cmap=ListedColormap(["#edb84b", "#488b70"]), vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(29), qp.FEATURES, fontsize=9)
    ax.set_xticks(range(31), known.site, rotation=90, fontsize=8)
    for tick, batch in zip(ax.get_xticklabels(), known.batch): tick.set_color(COLORS[batch])
    ax.set_title("Observed eligible numerical inputs after dependency gating (31 known sites)", loc="left")
    ax.legend(handles=[Patch(facecolor="#488b70", label="observed input; expert validity unconfirmed"),
                       Patch(facecolor="#edb84b", label="unavailable to model; fold median baseline used")],
              loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(assets / "coverage.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, variant in zip(axes, qp.VARIANTS):
        conf = pd.read_csv(OUT / f"confusion_{variant}.csv", index_col=0).to_numpy()
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=17)
        for i in range(3):
            for j in range(3): ax.text(j, i, str(conf[i,j]), ha="center", va="center", color="white" if conf[i,j] > 9 else "black")
        ax.set_xticks(range(3), ["B1", "B2", "B3"]); ax.set_yticks(range(3), ["B1", "B2", "B3"])
        ax.set(xlabel="predicted", ylabel="true", title=variant.replace("_", " "))
    fig.suptitle("Known-site LOO confusion counts; all four procedures use the same folds", x=.04, ha="left")
    fig.tight_layout(); fig.savefig(assets / "confusions.png", dpi=130); plt.close(fig)

    _, cat = load_runtime(ROOT / "analysis/submission_v2/freeze")
    from polaron_qc import features as feat
    examples = [("Dataset/Batch_1", "4ih2ggld", "known B1 / low contrast"),
                ("Hackathon-Polaron-test", "3e122cbj", "labelled B2 / low contrast"),
                ("Hackathon-Polaron-test", "fn0mhxef", "labelled B1 / ordinary"),
                ("Dataset/Batch_3", "71vgq3fw", "known B3 / raised black level"),
                ("Hackathon-Polaron-test", "xrv9xvzb", "labelled B3 / BSE stretching")]
    all_sites = pd.concat([known, pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/features.csv")]).set_index("site")
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 16))
    image_evidence = []
    for i, (folder, site, label) in enumerate(examples):
        paths = {d:Path(feat._image_path(ROOT / folder, site, d)) for d in ("BSE", "ETD", "Inlens")}
        raw = {d:feat._read_gray(path) for d,path in paths.items()}
        top, bottom = feat.bright_bands(raw["BSE"])
        trimmed = {d:a[top:len(a)-bottom if bottom else None] for d,a in raw.items()}
        row = all_sites.loc[site]
        pore, bright, _ = feat.segment(trimmed["BSE"], row.th_lo, row.th_hi)
        np.testing.assert_allclose([pore.mean(), bright.mean()], [row.pore_frac, row.bright_frac], atol=1e-12)
        inner_p = ndi.binary_erosion(bright, iterations=6)
        inner_g = ndi.binary_erosion(~pore & ~bright, iterations=8)
        joint = inner_p | inner_g
        h, w = pore.shape; side = min(512, h, w); y, x = (h-side)//2, (w-side)//2
        sl = np.s_[y:y+side, x:x+side]
        for j,d in enumerate(["BSE", "ETD", "Inlens"]):
            ax = axes[i,j]; ax.imshow(trimmed[d][sl], cmap="gray", vmin=0, vmax=255)
            colour = np.zeros((side, side, 4))
            if d == "BSE":
                colour[bright[sl]] = (1, .75, 0, .55); colour[pore[sl]] = (0, .85, 1, .5)
            else:
                colour[joint[sl]] = (.85, .25, .8, .23); colour[inner_p[sl]] = (1, .75, 0, .5)
            ax.imshow(colour); ax.set_axis_off()
            ax.set_title(f"{site}: {d}" if j else f"{label}\n{site}: BSE phase masks", fontsize=9)
        image_evidence.append(dict(site=site, label=label, review_status="unreviewed", raw_crop_x=x, raw_crop_y=y+top,
                                   side=side, trim_top=top, trim_bottom=bottom,
                                   pore_frac_verified=float(pore.mean()), bright_frac_verified=float(bright.mean()),
                                   joint_interior_fraction=float(joint.mean()),
                                   raw_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}))
    fig.suptitle("Fixed central crops: BSE void cyan / bright gold; joint ETD/Inlens interiors magenta / bright interiors gold\nUnreviewed algorithm selections, not phase truth. All grayscale displays fixed at 0–255.", fontsize=10)
    fig.tight_layout(rect=(0,0,1,.965)); fig.savefig(assets / "boundary_examples.jpg", dpi=120); plt.close(fig)

    primary = summary.set_index("variant").loc["dependency_gate"]
    legacy = summary.set_index("variant").loc["legacy_matched"]
    frozen = pd.read_csv(ROOT / "analysis/submission_v2/freeze/evaluation/categoriser_summary.csv").query("family == 'material'").iloc[0]
    table = summary[["variant", "n_sites", "balanced_accuracy", "accuracy", "recall_Batch_1", "recall_Batch_2", "recall_Batch_3", "perm_p", "brier"]]
    brief = failure[["variant", "site", "true_batch", "predicted_batch", "top_model_score", "n_observed_inputs", "deletion_same_assignment"]]
    conclusion = ("Known-site balanced accuracy improves in the matched comparison." if delta["balanced_accuracy_delta"] > 0
                  else "The quality gate does not improve known-site balanced accuracy in the matched comparison.")
    recommendation = "Keep the dependency policy as an experimental safety candidate and preserve frozen v2 as the declared submission. Invalid measurements must not be described as reliable material evidence. No result here establishes cross-session accuracy or warrants automatic deployment; a separate version/decision is needed. Independent boundary annotations and specimen/session grouping remain the next validation gates."
    markdown = ["# E37S: classifier input-quality audit", "", conclusion, "",
                f"Dependency gating: known-site balanced accuracy {primary.balanced_accuracy:.6f}, versus matched legacy {legacy.balanced_accuracy:.6f}; delta {delta['balanced_accuracy_delta']:+.6f}. Frozen D52 v2 was {frozen.balanced_accuracy:.6f}; the matched audit uses different shared inner-fold allocation, so these baselines must remain distinct.", "",
                "[Protocol](protocol.md) was fixed after first-drop feedback, before these comparisons. The drop is development evidence for this policy. All 31 known sites remain included (7/7/17), including cracked Batch 3 reference sites. No height, explicit session statistic or quality indicator enters any 29-input material/appearance candidate. All appearance inputs retain their acquisition qualifications.", "",
                table.to_markdown(index=False, floatfmt=".4f"), "",
                f"Paired OOF descriptive site-resampling delta interval {delta['descriptive_paired_oof_site_interval']}; {delta['newly_correct']} newly correct and {delta['newly_wrong']} newly wrong, {delta['changed_assignments']} labels changed. Overlapping CV fits and unknown specimen dependence prevent a calibrated generalisation interpretation. Primary permutation p is {primary.perm_p:.6f} with 200 full nested site-label permutations; advantage permutation diagnostic {delta['advantage_perm_p']:.6f}. Other policy/control p-values are secondary diagnostics, not selectable winners.", "",
                "Bright-only failure invalidates 19 of 29 candidate inputs; raised-black-level pore failure invalidates 16. Both unknown/failing masks leave only the two raw-BSE appearance summaries. Neither remaining inputs nor their imputed baseline terms are automatically trustworthy. Missingness indicators are absent, but imputation patterns can still carry acquisition information, as the quality-flags-only control tests.", "",
                "ETD particle ridge coverage depends on the bright interior AND the dominant orientation estimated using joint phase interiors. ETD graphite/ridge summaries and Inlens gradient energy also inherit phase-mask selection. Inlens particle standard-deviation summaries depend on the bright interior. A flag shown beside a score does not disable its numerical influence in v2. This audit marks invalid inputs missing before all fitting/selection/scoring.", "",
                "## Labelled failure examples — development only", "", brief.to_markdown(index=False, floatfmt=".4f"), "",
                "These examples were read only after known-only evaluation and candidate fits were saved. Their labels did not select C, transformations or variants. Their correctness is not new held-back accuracy; all three labels were already seen before the audit. Deletion counts are training-set sensitivity, not independent votes or confidence. Exact logit tables identify imputed baseline terms separately from observed inputs.", "",
                "## Recommendation", "", recommendation, "",
                "## Evidence and review", "",
                "[Illustrated report](report.html) · [Dependency catalogue](output/feature_dependencies.csv) · [Known OOF scores](output/known_oof_predictions.csv) · [Coverage](output/feature_availability.csv) · [Quality subgroups](output/quality_subgroups.csv) · [Failure eligibility](output/failure_input_eligibility.csv) · [Exact logit terms](output/failure_logit_contrasts.csv) · [Legacy invalid-input influence](output/legacy_invalid_input_influence.csv) · [Known-stage receipt](output/known_stage_receipt.json) · [Complete receipt](output/receipt.json).", "",
                "Image selections are unreviewed algorithm masks. Full-site overlay fractions reproduce saved values; central crops have recorded raw coordinates and fixed display. Batch labels do not supply phase/instance truth. Safety regression checks cover rejected-value invariance, missing flags, downstream dependencies, fold-local medians, empty training columns and site/row renaming. Frozen submission artifacts and production QC remain unchanged. Review C01–C06/C09/C11/C14/C16–C18/C21/C27–C30/C32–C35S plus C36S extraction dependencies.", ""]
    (HERE / "findings.md").write_text("\n".join(markdown))
    sections = [("Known-site comparisons", table.to_html(index=False, float_format=lambda x:f"{x:.4f}", border=0)),
                ("Dependencies", dependencies.to_html(index=False, border=0)),
                ("Failure examples — feedback-informed development", brief.to_html(index=False, float_format=lambda x:f"{x:.4f}", border=0))]
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>E37S input-quality audit</title>
    <style>body{{font:16px/1.5 system-ui;margin:36px auto;max-width:1450px;padding:0 20px;color:#23323b}}h1,h2{{line-height:1.2}}.warning{{background:#fff3ce;padding:16px;border-left:5px solid #c89820}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto}}section{{margin:30px 0}}a{{color:#286aa5}}</style></head><body>
    <h1>Classifier input-quality audit</h1><p>E37S / D56S · 2026-10-04 · candidate {escape(qp.VERSION)}</p>
    <div class="warning"><b>Feedback-informed development.</b> The three labelled drop examples are no longer an untouched test for these policies. Known-site folds overlap; scores are uncalibrated and specimen/session independence is unresolved. Frozen v2 and QC thresholds are unchanged. Zero independently reviewed phase masks.</div>
    <p>{escape(conclusion)} Dependency-gated balanced accuracy {primary.balanced_accuracy:.3f}, matched legacy {legacy.balanced_accuracy:.3f}, original frozen v2 {frozen.balanced_accuracy:.3f}. The audit fixes shared folds using mask-independent image content. Reordering explains why matched legacy and original v2 can differ.</p>
    <p>{escape(recommendation)}</p><p><a href="findings.md">Detailed findings</a> · <a href="protocol.md">Fixed protocol</a> · <a href="output/receipt.json">Provenance</a></p>
    {''.join(f'<section><h2>{title}</h2><div class="scroll">{content}</div></section>' for title,content in sections)}
    <section><h2>Observed input coverage</h2><p>Low bright contrast disables 19 inputs; raised black level disables 16. No site is silently dropped. Fold-trained median baseline terms remain distinct from observed evidence; patterns of missing inputs may still encode acquisition.</p><img src="{image_uri(assets/'coverage.png')}" alt="Known-site eligible input coverage"></section>
    <section><h2>LOO confusion counts</h2><img src="{image_uri(assets/'confusions.png')}" alt="Four matched-fold confusion matrices"></section>
    <section><h2>Boundary and interior review examples</h2><p>Five purposive examples, fixed central 512 px crops. Gold: thresholded bright phase/interiors; cyan: thresholded void; magenta: joint measurement interiors. Algorithm selections are unreviewed. Their crop counts are coverage, not independent samples or accuracy labels.</p><img src="{image_uri(assets/'boundary_examples.jpg')}" alt="Known and labelled failure example phase masks and detector interiors"></section>
    </body></html>'''
    (HERE / "report.html").write_text(html)
    (HERE / "image_evidence.json").write_text(json.dumps(image_evidence, indent=2) + "\n")
    (HERE / "report_receipt.json").write_text(json.dumps(dict(experiment="E37S", input_receipt_sha256=sha(OUT / "receipt.json"),
        output_sha256={str(p.relative_to(HERE)):sha(p) for p in [HERE/'findings.md', HERE/'report.html', HERE/'image_evidence.json', *sorted(assets.glob('*'))]},
        raw_evidence_read_only=True, five_overlay_fractions_verified=True, crops_unreviewed=True), indent=2) + "\n")
    print("Built E37S findings and self-contained report with coverage, confusion counts and five boundary examples.")


if __name__ == "__main__": main()
