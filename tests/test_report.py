"""Fast tests for polaron_qc.report: rendering on a small synthetic result (no extraction, no raw images) and the
evidence helpers on synthetic arrays."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pytest
from polaron_qc import PRIMARY_KPIS
from polaron_qc import report
from polaron_qc.decision import Thresholds, CONSISTENT

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- synthetic result
def _sites(n, batch):
    df = pd.DataFrame({k: rng.normal(1.0, 0.1, n) for k in PRIMARY_KPIS})
    df["bright_max_d"] = rng.normal(300, 20, n); df["corr_len_px"] = rng.normal(20, 2, n)
    df["pore_elong"] = rng.normal(2.0, 0.2, n); df["graphite_frac"] = 1 - df.pore_frac * 0.1 - df.bright_frac * 0.05
    df["th_lo"] = 40.0; df["th_hi"] = 100.0; df["H"] = 2000; df["bright_sep"] = 60.0
    df.insert(0, "site", [f"{batch[-1]}s{i}" for i in range(n)]); df.insert(0, "batch", batch)
    df["bright_low_contrast"] = False; df["grey_pore"] = False
    return df


def _compare(kpis):
    rows = []
    for k in kpis:
        prim = k in PRIMARY_KPIS
        rows.append(dict(kpi=k, is_primary=prim, n_ref_usable=17, n_batch_usable=7, n_ref_fallback=4 if k.startswith(("pore", "crack")) else 0,
                         n_batch_fallback=0, n_ref_excluded=0, n_batch_excluded=2 if k.startswith("bright") else 0,
                         ref_median=1.0, batch_median=1.05, ref_mad=0.1, shift_mad=0.5, ci_low=-0.3, ci_high=1.2, cliffs_delta=0.2,
                         p_perm=0.3, p_method="exact", n_perm=346104, p_holm=0.9 if prim else np.nan, p_reported=0.9 if prim else 0.3,
                         flag_alpha=False, direction="none"))
    return pd.DataFrame(rows)


def synthetic_result(verdict_text=CONSISTENT, localized="none"):
    th = Thresholds()
    ref, bat = _sites(17, "Batch_3"), _sites(7, "Batch_X")
    kpis = PRIMARY_KPIS + ["bright_max_d", "corr_len_px", "pore_elong"]
    cmp_ = _compare(kpis)
    mdc = {k: dict(mdc_mad=1.75, mdc_abs=0.175, n_incoming=7, n_sim_used=400, design="7 sites drawn without replacement vs remaining", ref_mad=0.1)
           for k in PRIMARY_KPIS}
    drift = pd.DataFrame(dict(site=list(ref.site) + list(bat.site), group=["reference"] * 17 + ["batch"] * 7,
                              distance=rng.uniform(0.5, 3, 24), percentile_in_ref=rng.uniform(0, 100, 24),
                              top_driver="pore_frac", top_driver_z=0.5))
    local = {k: dict(table=pd.DataFrame(dict(site=bat.site, value=bat[k], ref_max=1.3, exceeds=False, margin_in_mad=-1.0, credible_severity=False)),
                     ref_max=1.3, ref_mad=0.1, n_ref=10, n_batch=7, n_exceed=0, n_credible=0, iid_flag_probability=7 / 17, margin_mad=1.0)
             for k in PRIMARY_KPIS}
    per_kpi = {k: dict(shift_mad=0.5, ci=(-0.3, 1.2), p_holm=0.9, n_ref_usable=17, n_batch_usable=7, significant=False, direction="none",
                       consistency_share=0.0, n_beyond=0, usable_ok=True) for k in PRIMARY_KPIS}
    a = dict(i_beyond_null=False, energy_p=0.4, energy_stat=0.1, drivers=[], per_kpi=per_kpi, abstained_kpis=[], ii_carried_by_primary=False,
             iii_consistent=False, iv_classifier_corroborates=None, all_usable=True)
    b = dict(flags=[], credible=[], credible_pending_review=[], n_sites_flagged=0, n_sites_credible=0, n_sites_pending=0, max_severity_mad=np.nan)
    verdict = dict(verdict=verdict_text, reason="no drift beyond the reference null on primary KPIs and no credible localized anomaly",
                   outcome_columns=dict(drift_alert=False, localized=localized, quality_abstention=False), drivers=[],
                   attenuation=dict(stratified=np.nan, adjusted=np.nan), what_would_move_it="would become 'investigate — drift' if pore_frac crossed α",
                   thresholds_hash=th.hash(), provenance=th.provenance)
    stability = dict(share=1.0, n_runs=7, full_verdict=verdict_text, method="leave-one-site-out",
                     runs=[dict(left_out=s, verdict=verdict_text, outcome_columns=verdict["outcome_columns"]) for s in bat.site])
    flags = pd.DataFrame(dict(batch="Batch_X", site=bat.site, contrast_stretched_bse=False, contrast_stretched_any=True, bse_empty_bin_frac=0.1, bse_p1=3.0,
                              raised_black_level=False, bright_low_contrast=False, grey_pore=False, cracked=False, bright_sep=60.0, band_top=0, band_bottom=0,
                              H=2000, acquisition_group="ordinary"))
    meta = dict(reference="Batch_3", batch="Batch_X", reference_dir=None, batch_dir=None, n_sites_ref=17, n_sites_batch=7, timestamp="2026-10-03T12:00:00",
                thresholds_hash=th.hash(), provenance=th.provenance,
                config=dict(report.DEFAULT_CONFIG, thresholds_resolved=dict(severity_margin_mad=1.0)), config_hash="abc123", git_describe=None,
                feature_version="1.0.1", runtime_s=1.0, primary_kpis=list(PRIMARY_KPIS), trusted_kpis=kpis, notes=["synthetic"])
    limits = dict(usable_n={k: (17, 7) for k in PRIMARY_KPIS}, excluded={}, fallback={}, specimen_independence="unconfirmed",
                  reference_heterogeneity=dict(grey_pore=["g1"], cracked=["c1"], ordinary_n=10), mdc_range_mad=(1.5, 2.5),
                  scale="2-D sections, 25 nm/px nominal, chemistry unconfirmed", seven_v_seven="not applicable", notes=["synthetic"])
    return dict(meta=meta, sites_ref=ref, sites_batch=bat, images_ref=None, images_batch=None, patches_batch=None, flags_ref=flags, flags_batch=flags,
                ordinary_ref_sites=list(ref.site[:10]), compare=cmp_, energy=dict(statistic=0.1, p=0.4, n_perm=5000), mdc=mdc, drift=drift, local=local,
                check_a=a, check_b=b, abstention=dict(abstain=False, reasons=[]), verdict=verdict, stability=stability,
                c2st_material=None, c2st_flag_inclusive=None, c2st_extra={}, novelty=None, physics=None, acquisition=None, limits=limits,
                KPI_TRUST=dict(report.KPI_TRUST))


def test_render_report_synthetic(tmp_path):
    res = synthetic_result()
    out = report.render_report(res, str(tmp_path / "qc_Batch_X.html"))
    html = open(out, encoding="utf-8").read()
    low = html.lower()
    assert CONSISTENT in html
    for col in ("Drift alert", "Localized anomaly", "Quality abstention"):
        assert col in html
    assert "MDC" in html and "decision stability" in low
    assert "developed using exploratory analysis of batches 1–3; frozen before the unseen batch arrived" in low
    assert "confidence" not in low
    assert "instrument share" not in low
    # "accept" must not appear as a verdict (we check the verdict card, and that no verdict text contains it)
    card = re.search(r'<section class="card verdict[^"]*".*?</section>', html, re.S).group(0).lower()
    assert "accept" not in card
    # placeholders for missing optional blocks are rendered, not crashes
    assert "not available in this run" in low   # acquisition sensitivity block placeholder (reject withheld without it)
    assert "Evidence images not rendered" in html  # no raw images in the synthetic result


def test_kpi_trust_covers_primary():
    for k in PRIMARY_KPIS:
        assert k in report.KPI_TRUST and report.KPI_TRUST[k].startswith("high")
    # no flag or confounded KPI may sit in the trusted list (checklist C21)
    for k in report.TRUSTED_KPIS:
        assert not report.KPI_TRUST[k].startswith(("flag", "confounded", "diagnostic"))


# ----------------------------------------------------------------------------- evidence helpers
def _synthetic_bse():
    """Graphite-grey image with a long thin dark void (major axis ~600 px) and a short one (~100 px)."""
    img = np.full((400, 900), 120, np.uint8)
    img[180:200, 100:700] = 0        # long void: 600 px long
    img[300:320, 100:200] = 0        # short void: 100 px long
    img[50:90, 400:440] = 230        # one bright particle (40 x 40 = 1600 px)
    return img


def test_paint_crack_voids_only_long_components():
    img = _synthetic_bse()
    rgb = report.paint_crack_voids(img, th_lo=50, th_hi=200, major_axis_px=500)
    assert rgb.shape == img.shape + (3,) and rgb.dtype == np.uint8
    red = (rgb[..., 0] == 227) & (rgb[..., 1] == 73) & (rgb[..., 2] == 72)
    assert red[190, 400]            # inside the long void
    assert not red[310, 150]        # short void untouched
    assert not red[100, 100]        # background untouched
    # painted pixels are exactly the crack-void mask of the segmentation (long component only)
    from polaron_qc.features import segment
    pore, _, _ = segment(img, 50, 200)
    mask = report.crack_void_mask(pore, 500)
    assert np.array_equal(red, mask)
    assert mask[190, 400] and not mask[310, 150]
    # a lower cutoff paints both
    both = report.crack_void_mask(pore, 50)
    assert both[310, 150]


def test_outline_bright_marks_edge_not_interior():
    img = _synthetic_bse()
    rgb = report.outline_bright(img, th_lo=50, th_hi=200)
    yellow = (rgb[..., 0] == 255) & (rgb[..., 1] == 209) & (rgb[..., 2] == 102)
    assert yellow[50:90, 400:440].any()
    assert not yellow[70, 420]      # interior of the particle keeps its grey value
    assert rgb[70, 420, 0] == 230


def test_novelty_overlay_places_patches_on_the_right_rows():
    bse = np.full((1100, 512), 100, np.uint8)
    df = pd.DataFrame(dict(y0=[0, 512], x0=[0, 0], novelty=[0.0, 1.0]))
    rgb = report.novelty_overlay(bse, df, patch=512, alpha=0.5)
    assert rgb.shape == (1100, 512, 3)
    top, bottom, outside = rgb[100, 100].astype(int), rgb[600, 100].astype(int), rgb[1050, 100].astype(int)
    assert np.array_equal(outside, [100, 100, 100])         # rows beyond the two patches are untouched
    assert not np.array_equal(top, [100, 100, 100])         # both patches are tinted ...
    assert not np.array_equal(bottom, [100, 100, 100])
    assert not np.array_equal(top, bottom)                  # ... with different colours (novelty 0 vs 1)
    # the patch boundary is exactly at row 512
    assert np.array_equal(rgb[511, 100], rgb[100, 100]) and np.array_equal(rgb[512, 100], rgb[600, 100])


def test_fig_and_array_to_b64_downsample():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3)); ax.plot([0, 1])
    s = report.fig_to_b64(fig, width_px=600)
    assert s.startswith("data:image/jpeg;base64,")
    import base64, io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1])))
    assert im.width <= 600
    s2 = report.array_to_b64(np.zeros((100, 3000), np.uint8), width_px=1100)
    im2 = Image.open(io.BytesIO(base64.b64decode(s2.split(",", 1)[1])))
    assert im2.width == 1100


def test_derive_flags_rules():
    sites = _sites(2, "Batch_Y")
    images = pd.DataFrame([dict(batch="Batch_Y", site=sites.site[0], det="BSE", H=2000, mean=80, p1=15, p50=80, p99=200, std=30, gray_levels=120, empty_bin_frac=0.3, band_top=20, band_bottom=0),
                           dict(batch="Batch_Y", site=sites.site[1], det="BSE", H=2000, mean=80, p1=2, p50=80, p99=200, std=30, gray_levels=250, empty_bin_frac=0.05, band_top=0, band_bottom=0),
                           dict(batch="Batch_Y", site=sites.site[1], det="Inlens", H=2000, mean=80, p1=2, p50=80, p99=200, std=30, gray_levels=90, empty_bin_frac=0.6, band_top=0, band_bottom=0)])
    fl = report.derive_flags(sites, images)
    f = fl.set_index("site")
    s0, s1 = sites.site[0], sites.site[1]
    assert f.loc[s0, "contrast_stretched_bse"] and f.loc[s0, "raised_black_level"] and f.loc[s0, "band_rows"] == 20
    assert not f.loc[s1, "contrast_stretched_bse"] and f.loc[s1, "contrast_stretched_any"] and not f.loc[s1, "raised_black_level"]
    assert "acquisition" in fl.attrs["status"] and fl.attrs["config_mismatch"] == []
    # E21 P1: the grey-pore flag is data-derived on NEW site ids (raised black level), not a known-ID lookup
    assert bool(f.loc[s0, "grey_pore"]) and f.loc[s0, "grey_pore_source"] == "data" and f.loc[s0, "acquisition_group"] == "grey_pore"
    assert not f.loc[s1, "grey_pore"] and f.loc[s1, "acquisition_group"] == "ordinary"
    assert (f["H"] == 2000).all() and not f["cracked"].any()


def test_apply_derived_flags_reaches_stats_and_decision():
    """E21 P1 end to end on synthetic tables: five new sites with BSE p1 = 24 → grey_pore on all five → fallback count,
    Check B reliability note and quality abstention all see it."""
    from polaron_qc import stats, decision
    sites = _sites(5, "Batch_U")
    images = pd.DataFrame([dict(batch="Batch_U", site=s, det=d, H=1800, p1=24 if d == "BSE" else 0, p50=100, std=20, empty_bin_frac=0.0, band_top=0, band_bottom=0)
                           for s in sites.site for d in ("BSE", "ETD", "Inlens")])
    flags = report.derive_flags(sites, images)
    assert int(flags.grey_pore.sum()) == 5 and int(flags.raised_black_level.sum()) == 5
    merged = report.apply_derived_flags(sites, flags)
    assert merged.grey_pore.all() and (merged.acquisition_group == "grey_pore").all()
    assert stats.usable_n(merged, "pore_frac")["n_fallback"] == 5 and stats.usable_n(merged, "pore_frac")["n_usable"] == 5
    assert decision.quality_abstention(merged, {"all_usable": True})["abstain"] is True
    tab = pd.DataFrame([dict(site=merged.site[0], value=2.0, ref_max=1.3, exceeds=True, margin_in_mad=2.4)])
    b = decision.check_b({"crack_frac": tab}, merged)
    assert b["flags"][0]["measurement_note"].startswith("fallback") and b["n_sites_pending"] == 1   # soft flag: still pending
    # existing True values are kept (OR), never cleared
    sites2 = sites.copy(); sites2.loc[0, "bright_low_contrast"] = True
    assert report.apply_derived_flags(sites2, flags).bright_low_contrast.tolist() == [True, False, False, False, False]


def test_material_c2st_in_pipeline_shape():
    from polaron_qc import MATERIAL_KPIS
    rng2 = np.random.default_rng(5)
    def tbl(n, batch):
        df = pd.DataFrame({k: rng2.normal(1.0, 0.1, n) for k in MATERIAL_KPIS}); df.insert(0, "site", [f"{batch}_{i}" for i in range(n)]); df.insert(0, "batch", batch)
        return df
    ref, bat = tbl(12, "R"), tbl(6, "B"); bat.loc[0, "pore_d50"] = np.nan
    d = report.material_c2st(ref, bat, n_perm=20, seed=0)
    assert d["kpis"] == list(MATERIAL_KPIS) and d["n_sites_ref"] == 12 and d["n_sites_batch"] == 5 and d["n_dropped_nan"] == (0, 1)
    assert 0 <= d["p"] <= 1 and d["variant"].endswith("in_pipeline") and "in-pipeline" in d["p_method"] and d["flag_inclusive"] is False
    assert list(d["coef"].columns[:2]) == ["feature", "coef"]


def test_render_abstention_stability_wording_and_seed_sensitivity(tmp_path):
    res = synthetic_result(verdict_text="investigate — batch-wide drift")
    res["abstention"] = dict(abstain=True, reasons=["only 3 sites in batch (< 5)"])
    res["verdict"]["outcome_columns"]["quality_abstention"] = True
    res["c2st_material"] = dict(auc=0.59, null_band=[0.2, 0.8], p=0.30, n_perm=200, p_method="mc", cv="StratifiedGroupKFold(5)", n_sites_ref=17, n_sites_batch=7,
                                coef=pd.DataFrame(columns=["feature", "coef", "sign", "selected", "selection_stability"]), per_site_scores=pd.DataFrame(), kpis=["pore_frac"],
                                variant="material_all_sites_in_pipeline", key="k", flag_inclusive=False,
                                seed_sensitivity=dict(n_seeds=4, seeds=[0, 1, 2, 3], n_perm_extra=100, auc_min=0.34, auc_max=0.59, p_min=0.30, p_max=0.73, corroborates_all=False, corroborates_any=False))
    html = open(report.render_report(res, str(tmp_path / "qc.html")), encoding="utf-8").read()
    assert "not meaningful under a quality abstention" in html
    assert "Seed sensitivity" in html and "0.34–0.59" in html and "on no seed" in html


def test_render_report_with_acquisition_views_and_infeasible_mdc(tmp_path):
    res = synthetic_result()
    e = dict(statistic=0.2, p=0.3)
    res["acquisition"] = dict(unadjusted=dict(compare=None, energy=e), stratified=dict(energy=dict(statistic=0.15, p=0.4), n_ref_kept=10, n_batch_kept=7),
                              adjusted=dict(energy_adjusted=dict(statistic=0.3, p=0.2), n_ref=17, n_batch=7),
                              attenuation=dict(stratified=0.25, adjusted=-0.5, available=True), note="sensitivity analysis")
    res["verdict"]["attenuation"] = dict(stratified=0.25, adjusted=-0.5, available=True)
    res["mdc"]["crack_frac"] = dict(mdc_mad=np.nan, mdc_abs=np.nan, n_incoming=18, n_sim_used=0, feasible=False,
                                     reason="split design needs >= 3 remaining reference sites: n_incoming = 18 > n_ref - 3 = 14", design="not available")
    res["check_b"]["flags"] = [dict(site="Xs1", kpi="crack_frac", value=2.0, ref_max=1.3, margin_in_mad=2.4, severity_ok=True, measurement_reliable=True,
                                    measurement_note="ok", image_reviewed=False, review_status="refuted", promotable=True)]
    res["check_b"]["refuted"] = list(res["check_b"]["flags"])
    res["stability"]["refit_per_fold"] = ["compare", "material-only classifier", "acquisition views / attenuation"]; res["stability"]["held_fixed"] = []
    html = open(report.render_report(res, str(tmp_path / "qc.html")), encoding="utf-8").read()
    assert "Attenuation" in html and "+0.25" in html and "−0.50" in html and "drift reject is permitted" in html
    assert "not available</b>" in html and "split design" in html
    assert "refuted by image review (closed)" in html and ">refuted<" in html
    assert "Refit per fold" in html and "held fixed" not in html.split("Refit per fold")[1][:400]


def test_cli_passes_cache_dir_reviews_and_images(monkeypatch, tmp_path):
    """The one drop command: --cache-dir reaches build_result (cold-cache rehearsals, E25), --review becomes
    config['image_reviewed'], --no-images reaches the renderer, and the default cache dir is unchanged."""
    calls = {}

    def fake_build(ref, bat, config=None, cache_dir="analysis_cache/features", **kw):
        calls.update(ref=ref, bat=bat, config=config, cache_dir=cache_dir)
        return {"meta": {}}

    def fake_render(res, out, with_images=True):
        open(out, "w").write("<html/>"); calls.update(out=out, with_images=with_images); return out

    monkeypatch.setattr(report, "build_result", fake_build)
    monkeypatch.setattr(report, "render_report", fake_render)
    monkeypatch.setattr(report, "summary", lambda res: {"verdict": "x", "n": 1})
    out, summ = str(tmp_path / "qc.html"), str(tmp_path / "qc.json")
    path = report.main(["Dataset/Batch_3", "_fixtures/Batch_R", out, "--cache-dir", str(tmp_path / "fc"), "--summary", summ,
                        "--review", "abc:crack_frac=no", "--review", "def:pore_max_d=yes", "--no-images"])
    assert path == out and calls["out"] == out and calls["with_images"] is False
    assert calls["cache_dir"] == str(tmp_path / "fc") and calls["ref"] == "Dataset/Batch_3" and calls["bat"] == "_fixtures/Batch_R"
    assert calls["config"] == {"image_reviewed": {("abc", "crack_frac"): False, ("def", "pore_max_d"): True}}
    import json
    assert json.load(open(summ)) == {"verdict": "x", "n": 1}
    report.main(["Dataset/Batch_3", "_fixtures/Batch_R", str(tmp_path / "qc2.html")])
    assert calls["cache_dir"] == "analysis_cache/features" and calls["config"] is None and calls["with_images"] is True
    assert not os.path.exists(str(tmp_path / "qc2.json"))


def test_config_hash_accepts_image_reviews_and_is_stable_without_them():
    """E25 rehearsal: `--review SITE:KPI=yes` crashed in _config_hash (tuple keys are not JSON keys) before any result was
    assembled, so README step 4 had never run end to end. Reviews must hash; a run without reviews must keep its hash."""
    base = {**report.DEFAULT_CONFIG, "thresholds_resolved": {"alpha": 0.05}}
    h0 = report._config_hash(base)
    assert h0 == report._config_hash({**base, "image_reviewed": None})
    h1 = report._config_hash({**base, "image_reviewed": {("c20a68de", "crack_frac"): True}})
    h2 = report._config_hash({**base, "image_reviewed": {("c20a68de", "crack_frac"): False}})
    assert len(h1) == 12 and h1 != h0 and h1 != h2            # a review is part of the run's configuration


def test_summary_is_plain_json_and_carries_the_verdict_contract():
    """summary() must serialise with the strict JSON encoder (no NaN, no numpy scalars) and expose the fields the drop
    record needs: verdict, three outcome columns, flags with source, MDC feasibility, local flags with review state, notes."""
    import json
    res = synthetic_result()
    res["mdc"]["crack_frac"] = dict(mdc_mad=np.nan, mdc_abs=np.nan, n_incoming=18, n_sim_used=0, feasible=False, reason="split design", design="n/a")
    res["check_b"]["flags"] = [dict(site="Xs1", kpi="crack_frac", value=np.float64(2.0), ref_max=1.3, margin_in_mad=2.4, severity_ok=np.bool_(True),
                                    measurement_reliable=True, measurement_note="ok", image_reviewed=False, review_status="refuted", promotable=True)]
    s = report.summary(res)
    txt = json.dumps(s, allow_nan=False)                       # raises on NaN / inf; TypeError on numpy types
    back = json.loads(txt)
    assert back["verdict"] == res["verdict"]["verdict"]
    assert set(back["outcome_columns"]) == {"drift_alert", "localized", "quality_abstention"}
    assert back["mdc"]["crack_frac"]["feasible"] is False and back["mdc"]["crack_frac"]["mdc_mad"] is None
    assert back["local_anomaly"]["flags"][0]["review_status"] == "refuted" and back["local_anomaly"]["flags"][0]["value"] == 2.0
    assert {"site", "acquisition_group"} <= set(back["flags_batch"][0]) and len(back["flags_batch"]) == res["meta"]["n_sites_batch"]
    assert set(back["usable_n"]) == set(PRIMARY_KPIS) and {"share", "n_runs"} <= set(back["stability"])
    assert back["meta"]["thresholds_hash"] == res["meta"]["thresholds_hash"] and isinstance(back["notes"], list)
