"""Post-fit E41S renderer adapter; preserves the as-run frozen builder source.

This changes only labels and layout of recorded evidence. No extraction, model,
score or primary choice is changed. The delivery records this source separately.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import build_report as frozen


def main() -> None:
    frozen.main()
    path = frozen.HERE / "report.html"
    page = path.read_text()
    predictions = pd.read_csv(frozen.OUT / "known_oof_predictions.csv")
    coverage = predictions.groupby("candidate", sort=False).agg(
        crops=("site", "size"), min_observed=("n_observed_inputs", "min"),
        median_observed=("n_observed_inputs", "median"),
        max_observed=("n_observed_inputs", "max")).reset_index()
    expected = frozen.table(coverage)
    if page.count(expected) != 1:
        raise ValueError("Frozen report coverage block is ambiguous or changed")
    numerical = coverage.loc[coverage.candidate.isin(["legacy_l1", *frozen.MAIN])].copy()
    controls = coverage.loc[coverage.candidate.isin(["quality_rf", "availability_rf"])].copy()
    numerical.candidate = numerical.candidate.map(frozen.LABELS)
    controls.candidate = controls.candidate.map(frozen.LABELS)
    numerical = numerical.rename(columns={"min_observed": "Minimum numerical inputs observed",
        "median_observed": "Median numerical inputs observed", "max_observed": "Maximum numerical inputs observed"})
    controls = controls.rename(columns={"min_observed": "Minimum finite diagnostic bits",
        "median_observed": "Median finite diagnostic bits", "max_observed": "Maximum finite diagnostic bits"})
    replacement = ("<h3>Numerical measurement inputs</h3>" + frozen.table(numerical) +
        "<h3>Control inputs: finite flags and indicators</h3>" + frozen.table(controls))
    page = page.replace(expected, replacement)
    old_caption = ("Observed-input counts describe measurement coverage and are never extra training samples. "
        "Imputed inputs are model assumptions, not observed geometry. The main crop count remains 31.")
    new_caption = ("The five measurement-model rows count observed numerical inputs after their declared gate "
        "(the ungated reference preserves its original inputs). The two controls count finite quality flags or "
        "availability indicators; those bits are not observed material geometry. Imputed numerical values are "
        "model assumptions, not measurements. No count creates extra training samples; n remains 31 crops.")
    if page.count(old_caption) != 1:
        raise ValueError("Frozen report coverage caption is ambiguous or changed")
    page = page.replace(old_caption, new_caption)
    note = ("<p class=\"small\">Post-fit delivery rendering separates numerical measurement coverage "
        "from finite control bits. It uses the preserved as-run builder plus "
        "<a href=\"delivery_report.py\">this separately recorded renderer adapter</a>; "
        "no feature definition, fit, score or model choice changes.</p>")
    page = page.replace("</main></body></html>", note + "</main></body></html>")
    path.write_text(page)
    print("Delivery report SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())
    print("Delivery renderer SHA-256:", hashlib.sha256((frozen.HERE / "delivery_report.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
