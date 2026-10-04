"""E41S fixed, unvalidated 2-D geometry inputs; never changes frozen QC.

Raw extraction does not read labels, prediction outcomes or expert annotations.
Only ``prepare`` applies the same fail-closed phase-quality policy as E37S.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, regionprops_table

from polaron_qc import features
from analysis.quality_policy_audit import policy as qp
from analysis.morphology.benchmark_methods import local_width
from analysis.morphology.k_pilot.descriptors import shape_summary, phase_summary
from analysis.ml_options.e_graph.graph import build_graph

FEATURES = [
    'bright_aspect_count_iqr', 'bright_circularity_count_iqr', 'bright_solidity_count_q10',
    'void_local_width_d50_px', 'void_local_width_d90_px', 'graph_edge_length_iqr_ratio',
    'local_bright_std_512px', 'bright_pore_crosscorr_xy_contrast_256px',
]
BRIGHT = FEATURES[:3] + [FEATURES[5], FEATURES[6]]
PORE = FEATURES[3:5]
JOINT = [FEATURES[7]]
VARIANTS = {'nominal': (0, 50), 'threshold_minus5': (-5, 50),
            'threshold_plus5': (5, 50), 'floor100': (0, 100)}
VERSION = 'E41S-geometry-v1'
PROPS = ('label', 'area', 'major_axis_length', 'minor_axis_length', 'centroid',
         'bbox', 'solidity', 'perimeter')
DEFINITIONS = {
    FEATURES[0]: 'Linear Q75-Q25 of major/minor-axis ratio, equal weight per eligible bright component.',
    FEATURES[1]: 'Linear Q75-Q25 of 4*pi*area/perimeter^2, equal weight per eligible bright component; not clipped at one.',
    FEATURES[2]: 'Linear Q10 of raster solidity, equal weight per eligible bright component.',
    FEATURES[3]: 'Median 2*Euclidean distance at deterministic medial-axis pixels of unclipped void components >=30 px^2; centreline-pixel pooled.',
    FEATURES[4]: 'Linear Q90 of those centreline local-width samples.',
    FEATURES[5]: 'Linear edge-length IQR / median distance in the unique undirected union of retained bright centroids\' 3 nearest distinct neighbours.',
    FEATURES[6]: 'Population SD of original bright-mask area fractions over complete 512x512-px non-overlapping top-left-anchored windows; partial windows excluded.',
    FEATURES[7]: 'Mean binary bright-to-void Pearson correlation at signed +/-256-px x lags minus mean at signed +/-256-px y lags; pair-specific finite-domain marginals.',
}


def _validate_frame(frame, with_variant=False):
    """Reject ambiguous variant joins rather than silently choosing first rows."""
    keys = ['batch', 'site'] if 'batch' in frame else ['site']
    if 'site' not in frame or frame.site.isna().any():
        raise ValueError('Nonmissing site identifiers required for one-row-per-crop join')
    if with_variant:
        if 'variant' not in frame:
            raise ValueError('Explicit variant required')
        keys += ['variant']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate crop/variant measurements')
    missing = set(FEATURES) - set(frame)
    if missing:
        raise ValueError(f'Missing fixed geometry measurements: {sorted(missing)}')


def nominal(frame):
    """Explicitly filter a variant table; duplicate nominal measurements fail."""
    if 'variant' not in frame:
        raise ValueError('Explicit variant column required')
    _validate_frame(frame, with_variant=True)
    out = frame.loc[frame.variant.eq('nominal')].copy()
    _validate_frame(out)
    return out


COVERAGE = {
    **{k: [('shape_objects_eligible', 20)] for k in FEATURES[:3]},
    **{k: [('n_retained_voids', 1), ('n_width_samples', 1)] for k in PORE},
    FEATURES[5]: [('graph_nodes', 10), ('graph_edges', 10)],
    FEATURES[6]: [('neighbourhood_windows_512px', 2), ('neighbourhood_windows_bright_present_512px', 1)],
    FEATURES[7]: [('crosscorr_signed_domains', 4), ('crosscorr_min_pair_pixels', 10000),
        ('crosscorr_min_source_phase_pixels', 100), ('crosscorr_min_target_phase_pixels', 100),
        ('crosscorr_min_source_complement_pixels', 100), ('crosscorr_min_target_complement_pixels', 100)],
}


def coverage_reason(row, feature):
    """Persisted coverage must certify the raw finite value; absence fails closed."""
    for field, minimum in COVERAGE[feature]:
        value = pd.to_numeric(pd.Series([row.get(field, np.nan)]), errors='coerce').iloc[0]
        if not np.isfinite(value) or value < 0:
            return 'coverage_unknown:' + field
        if value < minimum:
            return 'coverage_below_minimum:' + field
    if feature == FEATURES[7] and float(row['crosscorr_signed_domains']) != 4:
        return 'coverage_not_four_signed_domains'
    return ''


def prepare(frame):
    """Return fixed8 matrix, names, per-cell availability and phase-quality flags.

    No flag, coverage count, crop identifier, height or session statistic is a
    predictor. Imputation belongs exclusively inside the downstream training fold.
    Unknown bright/pore quality fails closed; rejected numerical values never
    enter the matrix. Raw columns remain unchanged for diagnostic inspection.
    """
    _validate_frame(frame)
    q = qp.quality(frame)
    X = frame[FEATURES].to_numpy(float).copy()
    audits = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        for j, feature in enumerate(FEATURES):
            bright_dep = feature in BRIGHT + JOINT
            pore_dep = feature in PORE + JOINT
            finite = bool(np.isfinite(X[i, j]))
            reasons = []
            if bright_dep and q.iloc[i].bright_bad:
                reasons.append(q.iloc[i].bright_reason)
            if pore_dep and q.iloc[i].pore_bad:
                reasons.append(q.iloc[i].pore_reason)
            coverage = coverage_reason(row, feature)
            if coverage:
                reasons.append(coverage)
            if not finite:
                raw_reason = row.get(feature + '_unavailable_reason', '')
                reasons.append(str(raw_reason) if pd.notna(raw_reason) and raw_reason else 'measurement_unavailable')
            if reasons:
                X[i, j] = np.nan
            audits.append(dict(batch=str(row.get('batch', '')), site=str(row.site),
                cohort=str(row.get('cohort', '')), feature=feature, raw_value=float(row[feature]),
                raw_finite=finite, bright_dependency=bright_dep, pore_dependency=pore_dep,
                coverage_eligible=not bool(coverage),
                policy_eligible=not ((bright_dep and q.iloc[i].bright_bad) or (pore_dep and q.iloc[i].pore_bad)),
                observed_model_input=bool(np.isfinite(X[i, j])), reason='|'.join(reasons)))
    return X, list(FEATURES), pd.DataFrame(audits), q


def component_table(mask):
    """Same raster properties/8-connectivity as E18/E26G, prefiltered >=50 px²."""
    t = pd.DataFrame(regionprops_table(label(mask, connectivity=2), properties=PROPS))
    t = t.loc[t.area.ge(50)].copy()
    h, w = mask.shape
    t['touches_edge'] = (t['bbox-0'].eq(0) | t['bbox-1'].eq(0) |
                         t['bbox-2'].eq(h) | t['bbox-3'].eq(w))
    return t


def local_bright_spread(bright, size=512):
    """Exact E25 raster-window SD convention, without unrelated neighbourhood work."""
    h, w = bright.shape
    ny, nx = h // size, w // size
    n = ny * nx
    fractions = bright[:ny*size, :nx*size].reshape(ny, size, nx, size).mean(axis=(1, 3)).ravel() if n else np.array([])
    reason = 'fewer_than_2_complete_windows' if n < 2 else ('no_bright_pixels_in_complete_windows' if not (fractions > 0).any() else '')
    value = float(np.std(fractions, ddof=0)) if not reason else np.nan
    return value, dict(neighbourhood_windows_512px=n,
        neighbourhood_window_coverage_512px=n*size*size/bright.size,
        neighbourhood_windows_bright_present_512px=int((fractions > 0).sum()),
        local_bright_std_512px_unavailable_reason=reason), fractions.reshape(ny, nx)


def extract_site(raw_bse, *, metadata=None, threshold_offset=0, area_floor=50,
                 thresholds=None, return_maps=False):
    """Extract fixed8 from one raw uint8 BSE image, independent of class/truth.

    Adaptive trim/thresholds/flags replay current extraction. An optional exact
    compatible nominal threshold pair can be supplied for historical parity.
    Threshold offsets move BOTH boundaries. Bright floor100 affects shape/graph
    only: width floor30 and raster-window/cross-correlation inputs do not change.
    """
    raw = np.asarray(raw_bse)
    if raw.ndim != 2 or raw.dtype != np.uint8 or not raw.size:
        raise ValueError('Nonempty 2-D uint8 raw BSE required')
    if threshold_offset not in (-5, 0, 5) or area_floor not in (50, 100):
        raise ValueError('Only preregistered offsets/floors are supported')
    top, bottom = features.bright_bands(raw)
    trimmed = features.trimmed(raw)
    if min(trimmed.shape) < 2:
        raise ValueError('Trim leaves insufficient field')
    sm = features.smooth_bse(trimmed)
    th = features.phase_thresholds(sm)
    if thresholds is not None:
        lo, hi = thresholds
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Invalid threshold pair')
        if not np.allclose([lo, hi], [th['th_lo'], th['th_hi']], atol=1e-8, rtol=1e-8):
            raise ValueError('Historical thresholds do not match current adaptive extraction')
        th['th_lo'], th['th_hi'] = float(lo), float(hi)
    pore = ndi.binary_opening(sm < th['th_lo'] + threshold_offset, iterations=1)
    bright = ndi.binary_opening(sm > th['th_hi'] + threshold_offset, iterations=1)
    nodes = component_table(bright)
    shapes, shape_nodes = shape_summary(nodes.loc[nodes.area.ge(area_floor)])
    graph, graph_nodes, edges = build_graph(nodes, area_floor)
    width, width_maps = local_width(pore, min_area=30, return_maps=True)
    phases, pairs = phase_summary(bright, pore)
    local, local_diagnostics, windows = local_bright_spread(bright)
    values = {k: (shapes | graph | width | phases).get(k, np.nan) for k in FEATURES}
    values[FEATURES[6]] = local
    diagnostic_keys = ['shape_objects_input', 'shape_objects_eligible', 'shape_objects_excluded',
        'shape_circularity_above_one', 'graph_nodes', 'graph_edges', 'graph_components_available',
        'graph_components_below_floor', 'graph_components_clipped', 'graph_zero_length_edges',
        'n_retained_voids', 'n_width_samples', 'retained_void_area_frac', 'excluded_void_area_frac']
    diagnostics = {k: (shapes | graph | width)[k] for k in diagnostic_keys}
    diagnostics.update(local_diagnostics)
    cp = pairs.loc[pairs.kind.eq('cross') & pairs.lag_px.abs().eq(256)]
    diagnostics.update(crosscorr_signed_domains=len(cp),
        crosscorr_min_pair_pixels=int(cp.pair_pixels.min()),
        crosscorr_min_source_phase_pixels=int(cp.source_phase_pixels.min()),
        crosscorr_min_target_phase_pixels=int(cp.target_phase_pixels.min()),
        crosscorr_min_source_complement_pixels=int((cp.pair_pixels-cp.source_phase_pixels).min()),
        crosscorr_min_target_complement_pixels=int((cp.pair_pixels-cp.target_phase_pixels).min()))
    for k in FEATURES[:3]:
        diagnostics[k+'_unavailable_reason'] = shapes['shape_unavailable_reason']
    for k in FEATURES[3:5]:
        diagnostics[k+'_unavailable_reason'] = '' if np.isfinite(width[k]) else 'no_retained_void_centreline_samples'
    for k in [FEATURES[5], FEATURES[7]]:
        diagnostics[k+'_unavailable_reason'] = (graph | phases)[k+'_unavailable_reason']
    # E37 pore-quality rule uses the same integer p1 as exported image_quality.
    header = dict(bright_low_contrast=bool(th['bright_low_contrast']),
        bse_p1=int(np.percentile(raw, 1)), grey_pore=bool(int(np.percentile(raw, 1)) > 10),
        th_lo=float(th['th_lo']), th_hi=float(th['th_hi']), threshold_offset=threshold_offset,
        area_floor_px2=area_floor, trim_top=int(top), trim_bottom=int(bottom),
        H_trimmed=trimmed.shape[0], W=trimmed.shape[1],
        pore_frac=float(pore.mean()), bright_frac=float(bright.mean()), version=VERSION)
    values.update(header)
    if metadata:
        # These are identifiers only, never used in masks, gates or predictors.
        for k in ('batch', 'site', 'cohort', 'variant'):
            if k in metadata:
                values[k] = metadata[k]
    if return_maps:
        return values, diagnostics, dict(raw_trimmed=trimmed, pore=pore, bright=bright,
            component_table=nodes, shape_nodes=shape_nodes, graph_nodes=graph_nodes,
            graph_edges=edges, width=width_maps, phase_pairs=pairs, bright_window_fractions=windows)
    return values, diagnostics
