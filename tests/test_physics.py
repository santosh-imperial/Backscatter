"""Fast tests for polaron_qc.physics on synthetic masks only (no TIFF access)."""
import os

import numpy as np
import pandas as pd
import pytest

from polaron_qc import physics as ph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "analysis_cache")


# ---------------------------------------------------------------- tortuosity
def test_tortuosity_straight_slot_is_one():
    m = np.zeros((400, 200), bool)
    m[:, 90:110] = True                      # 20-px wide vertical slot
    r = ph.section_tortuosity_index(m, n_lines=10)
    assert r["index"] == pytest.approx(1.0, abs=0.02)
    assert r["fraction_connected"] == 1.0
    assert r["caveats"] and r["claims_not_made"]


def test_tortuosity_zigzag_above_1p3():
    H, W = 480, 240
    m = np.zeros((H, W), bool)
    # staircase: vertical risers at alternating sides joined by horizontal runs
    step = 80
    y = 0
    left = True
    while y < H:
        y2 = min(y + step, H)
        if left:
            m[y:y2, 10:30] = True
        else:
            m[y:y2, W - 30:W - 10] = True
        m[max(y2 - 20, 0):y2, 10:W - 10] = True   # horizontal run at the bottom of each riser
        y = y2; left = not left
    r = ph.section_tortuosity_index(m, n_lines=10)
    assert r["fraction_connected"] == 1.0
    assert r["index"] > 1.3


def test_tortuosity_blocked_by_wall():
    m = np.ones((400, 200), bool)
    m[200:220, :] = False                    # solid wall across the full width
    r = ph.section_tortuosity_index(m, n_lines=20)
    assert r["fraction_connected"] == 0.0
    assert np.isnan(r["index"])
    assert "undefined" in r["statement"]
    assert np.isfinite(r["penalised_index"]) and r["penalised_index"] > 1.0


# ---------------------------------------------------------------- void geometry
def test_void_geometry_counts_crack_like():
    m = np.zeros((2000, 3000), bool)
    m[1000:1020, 1000:1600] = True           # 600 × 20 px horizontal void
    r = ph.void_geometry(m, strip_height_px=2000)
    assert r["n_crack_like"] == 1
    assert r["longest_void_px"] > 500
    assert 0.15 < r["columns_interrupted_frac"] < 0.25     # 600/3000 = 0.2
    assert r["delamination_index"] == pytest.approx(0.2, abs=0.01)
    # major_axis_length is the ellipse-equivalent axis: 2/sqrt(3) × rectangle length ≈ 693 px (same definition as notebook 01)
    assert r["longest_void_over_thickness"] == pytest.approx(600 * 2 / np.sqrt(3) / 2000, rel=0.02)
    assert r["crack_orientation_deg_median"] < 5
    assert r["statements"] and r["caveats"] and r["claims_not_made"]
    assert any("not evidence of electronic discontinuity" in s for s in r["statements"] + r["caveats"])


def test_void_geometry_ignores_short_void():
    m = np.zeros((2000, 3000), bool)
    m[1000:1020, 1000:1400] = True           # 400 × 20 px
    r = ph.void_geometry(m, strip_height_px=2000)
    assert r["n_crack_like"] == 0
    assert r["columns_interrupted_frac"] == 0.0
    assert r["delamination_index"] == 0.0
    assert r["n_voids"] == 1


def test_void_geometry_empty_mask():
    r = ph.void_geometry(np.zeros((100, 100), bool))
    assert r["n_crack_like"] == 0 and r["statements"] and r["caveats"]


# ---------------------------------------------------------------- stereology
def test_saltykov_conserves_count_order_of_magnitude():
    rng = np.random.default_rng(0)
    # sections of monodisperse spheres (D = 100): chord diameters 100·sqrt(1 − u²), u uniform
    u = rng.uniform(0, 1, 2000)
    d2 = 100 * np.sqrt(1 - u ** 2)
    out = ph.saltykov_unfold(d2, n_bins=12)
    assert len(out) == 12
    nv = out["n_3d_per_volume_relative"].to_numpy()
    assert np.isfinite(nv).all()
    implied = out.attrs["implied_n_2d_from_3d"]
    assert 0.3 * 2000 < implied < 3 * 2000
    # the largest class should dominate for monodisperse spheres
    assert np.argmax(nv) >= 10
    assert out.attrs["secondary"] is True and out.attrs["caveats"]


def test_saltykov_nan_safe_three_diameters():
    out = ph.saltykov_unfold([120.0, 80.0, 30.0], n_bins=12)
    assert len(out) == 12
    assert not np.isinf(out["n_3d_per_volume_relative"].fillna(0)).any()
    out1 = ph.saltykov_unfold([50.0], n_bins=12)
    assert out1["n_3d_per_volume_relative"].isna().all()
    out0 = ph.saltykov_unfold([], n_bins=12)
    assert out0["n_3d_per_volume_relative"].isna().all()
    q = ph.unfolded_quantiles(out0)
    assert all(np.isnan(v) for v in q.values())


def test_stereology_delesse_and_mass_fraction():
    row = dict(batch="B", site="s", pore_frac=0.10, bright_frac=0.05, graphite_frac=0.85)
    S = ph.stereology(row)
    assert S.vol_frac_pore.iloc[0] == 0.10 and S.vol_frac_bright.iloc[0] == 0.05
    mf = S.nominal_mass_frac_additive_if_Si.iloc[0]
    assert mf == pytest.approx(2.33 * 0.05 / (2.33 * 0.05 + 2.26 * 0.85))
    assert S.nominal_mass_frac_additive_if_SiOx.iloc[0] < mf
    assert bool(S.stereology_secondary.iloc[0]) is True
    assert S.attrs["caveats"] and S.attrs["claims_not_made"]
    assert "plate alignment" in S.attrs["column_notes"]["vol_frac_graphite"]
    # DataFrame input
    S2 = ph.stereology(pd.DataFrame([row, row]))
    assert len(S2) == 2


# ---------------------------------------------------------------- readings
def test_additive_size_reading():
    r = ph.additive_size_reading(dict(d10=20, d50=150, d90=300), dict(d10=21, d50=180, d90=360))
    assert r["direction"]["d50"] == "coarser" and r["direction"]["d10"] == "similar"
    assert r["ratios"]["d90"] == pytest.approx(1.2)
    assert "D90 coarser (×1.20)" in r["statement"]
    assert "do not determine later lithiation time" in r["statement"]
    assert "r2_ratio_d90" not in r
    assert any("bias may differ" in x for x in r["caveats"])
    assert r["claims_not_made"]
    with pytest.raises(ValueError):
        ph.additive_size_reading(dict(d50=1), dict(d50=1), basis="3d")


def test_additive_mechanics_reading():
    ref = dict(bright_frac=0.05, bright_d50=150, bright_low_contrast=False, etd_crack_density_particles=0.002)
    bat = dict(bright_frac=0.07, bright_d50=150, bright_low_contrast=False, etd_crack_density_particles=0.003)
    r = ph.additive_mechanics_reading(ref, bat)
    assert r["direction"] == "not inferred" and r["loading_direction"] == "more"
    assert r["ridge_coverage_batch"] == pytest.approx(0.003)
    assert "intact_share_proxy_batch" not in r
    assert r["caveats"] and r["claims_not_made"] and "not an intact-particle share" in r["statement"]
    low = dict(bat, bright_low_contrast=True)
    assert ph.additive_mechanics_reading(ref, low)["direction"].startswith("unreliable")
    less = dict(bright_frac=0.04, bright_d50=120, bright_low_contrast=False)
    assert ph.additive_mechanics_reading(ref, less)["loading_direction"] == "less"
    mixed = dict(bright_frac=0.07, bright_d50=120, bright_low_contrast=False)
    assert ph.additive_mechanics_reading(ref, mixed)["direction"] == "not inferred"
    coarser_only = ph.additive_mechanics_reading(ref, dict(ref, bright_d50=300))
    assert coarser_only["loading_direction"] == "similar"
    assert coarser_only["size_direction"] == "coarser" and coarser_only["direction"] == "not inferred"
    assert ph.additive_mechanics_reading(dict(ref, bright_frac=0), bat)["loading_direction"] == "not measurable"


def test_qualitative_statement_pattern_and_weight_one_is_empty():
    s = ph.qualitative_statement("pore_frac", -2.1, "lower", 0.09, 0.07)
    assert s == ("Macro-pore area fraction is lower than the reference (robust shift 2.1 MAD) → "
                 "less segmented 2-D void area; packing/wetting implications need independent validation, direction only.")
    assert ph.qualitative_statement("bright_low_contrast", 3.0, "higher", 0, 1) == ""
    assert ph.qualitative_statement("inlens_particle_texture", 3.0, "higher", 0, 1) == ""
    assert ph.qualitative_statement("bright_d50", 1.5, None, 150, 170).startswith("Bright-phase area-weighted D50 is higher")
    assert "%" not in ph.qualitative_statement("crack_frac", 2.0, "higher", 0.01, 0.05)


# ---------------------------------------------------------------- sanity checks
def test_sanity_checks():
    df = pd.DataFrame([dict(batch="B", site="a", pore_frac=0.08, bright_frac=0.05, graphite_frac=0.87, pore_elong=2.5, pore_mode_resolved=True, bright_low_contrast=False),
                       dict(batch="B", site="b", pore_frac=0.30, bright_frac=0.05, graphite_frac=0.64, pore_elong=1.0, pore_mode_resolved=False, bright_low_contrast=False)])
    C = ph.sanity_checks(df)
    assert C.fractions_sum_to_one.tolist() == [True, False]
    assert C.porosity_vs_typical.tolist() == ["below_typical", "within_typical"]
    assert "unresolved porosity" in C.porosity_note.iloc[0] and C.porosity_note.iloc[1] == ""
    assert C.anisotropy_index.iloc[0] == pytest.approx(1.5)
    assert C.attrs["caveats"] and C.attrs["claims_not_made"]
    sens = pd.DataFrame([dict(site="a", pore_frac_band=0.01, bright_frac_band=0.004)])
    C2 = ph.sanity_checks(df, sens)
    assert C2.pore_frac_band.iloc[0] == 0.01 and np.isnan(C2.pore_frac_band.iloc[1])


# ---------------------------------------------------------------- weights
def test_weights_cover_all_cache_columns():
    cols = set(pd.read_csv(os.path.join(CACHE, "site_features.csv"), nrows=1).columns)
    cols |= set(pd.read_csv(os.path.join(CACHE, "etd_inlens_features.csv"), nrows=1).columns)
    missing = sorted(c for c in cols if c not in ph.CONSEQUENCE_WEIGHTS)
    assert not missing, f"no consequence weight for: {missing}"
    assert set(ph.CONSEQUENCE_WEIGHTS.values()) <= {1, 2, 3}
    assert set(ph.CONSEQUENCE_RATIONALE) == set(ph.CONSEQUENCE_WEIGHTS)
    for k in ("crack_frac", "pore_max_d", "bright_frac", "bright_d50", "pore_frac"):
        assert ph.CONSEQUENCE_WEIGHTS[k] == 3
    assert ph.CONSEQUENCE_WEIGHTS["inlens_particle_texture"] == 1
    v = ph.weights_vector(["crack_frac", "pore_d50", "th_lo"])
    assert v.tolist() == [3.0, 2.0, 1.0]
    with pytest.warns(UserWarning):
        assert ph.weights_vector(["no_such_kpi"]).tolist() == [1.0]
    # every KPI with a physics reading has direction text for both directions
    for k, w in ph.CONSEQUENCE_WEIGHTS.items():
        if w >= 2:
            assert set(ph.CONSEQUENCE_TEXT[k]) == {"higher", "lower"}, k


def test_every_statement_function_returns_caveats():
    m = np.zeros((200, 200), bool); m[:, 50:60] = True
    outs = [ph.section_tortuosity_index(m, n_lines=3), ph.void_geometry(m),
            ph.additive_size_reading(dict(d50=1.0), dict(d50=1.1)),
            ph.additive_mechanics_reading(dict(bright_frac=0.05, bright_d50=100), dict(bright_frac=0.05, bright_d50=100))]
    for o in outs:
        assert len(o["caveats"]) > 0 and len(o["claims_not_made"]) > 0
    for df in (ph.stereology(dict(pore_frac=0.1, bright_frac=0.05)), ph.saltykov_unfold([1, 2, 3]),
               ph.sanity_checks(pd.DataFrame([dict(site="x", pore_frac=0.1, bright_frac=0.05)]))):
        assert len(df.attrs["caveats"]) > 0 and len(df.attrs["claims_not_made"]) > 0
