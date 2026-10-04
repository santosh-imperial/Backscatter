"""Fixed 2-D centroid proximity geometry; no physical-contact or QC claims."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

KEYS = ('graph_edge_length_median_px', 'graph_edge_length_iqr_ratio',
        'graph_edge_axial_strength', 'graph_log_area_assortativity')
DEFINITIONS = {
    KEYS[0]: ('px', 'Median Euclidean centroid distance across the unique undirected union of each retained node\'s three nearest distinct neighbours.'),
    KEYS[1]: ('ratio', 'Edge-distance interquartile range (linear quantiles) divided by median edge distance; unavailable for zero median.'),
    KEYS[2]: ('0–1', 'Unweighted absolute mean exp(2j theta) across positive-length undirected edges; theta uses image x/y axes.'),
    KEYS[3]: ('−1–1', 'Pearson correlation of natural-log component area at both endpoints, including both directed copies of every unique undirected edge; unavailable for zero endpoint variance.'),
}


def build_graph(components, area_floor=50):
    """Exclude clipped nodes; fix 3-NN topology with coordinate tie-breaking.

    Input component labels come from one full-frame mask. Sorting makes input
    row order irrelevant; labels break any remaining coincident-coordinate tie.
    Geometry below the pilot floor remains visible, with unavailable summaries.
    """
    required = {'label', 'area', 'centroid-0', 'centroid-1', 'touches_edge'}
    if not required <= set(components):
        raise ValueError('Missing component geometry columns')
    t = components.copy()
    numeric = t[['area', 'centroid-0', 'centroid-1']].to_numpy(float)
    if not np.isfinite(numeric).all() or (t.area <= 0).any() or t.label.duplicated().any():
        raise ValueError('Invalid/duplicate component geometry')
    if not t.touches_edge.isin([True, False]).all():
        raise ValueError('Unknown clipping state')
    retained = t[t.area >= area_floor]
    nodes = retained[retained.touches_edge.eq(False)].sort_values(
        ['centroid-0', 'centroid-1', 'label'], kind='stable').reset_index(drop=True)
    points = nodes[['centroid-0', 'centroid-1']].to_numpy(float)
    edges = set()
    if len(points) >= 2:
        tree = cKDTree(points)
        k = min(3, len(points) - 1)
        for i, p in enumerate(points):
            # Query k+1 because one candidate is this node, even if another
            # component has an identical centroid. Include all distance ties.
            dist, _ = tree.query(p, k=k+1)
            radius = np.nextafter(float(np.max(dist)), np.inf)
            candidates = [j for j in tree.query_ball_point(p, radius) if j != i]
            candidates.sort(key=lambda j: (float(np.sum((points[j]-p)**2)),
                                          points[j, 0], points[j, 1], int(nodes.iloc[j].label)))
            for j in candidates[:k]:
                edges.add(tuple(sorted((i, j))))
    ij = np.asarray(sorted(edges), dtype=int).reshape(-1, 2)
    if len(ij):
        delta = points[ij[:, 1]] - points[ij[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        angle = np.arctan2(delta[:, 0], delta[:, 1])
    else:
        length, angle = np.array([]), np.array([])
    edge_table = pd.DataFrame(dict(node_a=ij[:, 0], node_b=ij[:, 1],
                                   length_px=length, angle_rad=angle))
    metrics = {key: np.nan for key in KEYS}
    reasons = {key: 'coverage_below_10_nodes_or_10_edges' for key in KEYS}
    if len(nodes) >= 10 and len(ij) >= 10:
        med = float(np.median(length))
        metrics[KEYS[0]] = med
        reasons[KEYS[0]] = ''
        if med > 0:
            metrics[KEYS[1]] = float(np.diff(np.quantile(length, [.25, .75]))[0] / med)
            reasons[KEYS[1]] = ''
        else:
            reasons[KEYS[1]] = 'zero_median_edge_length'
        positive = length > 0
        if positive.any():
            metrics[KEYS[2]] = float(abs(np.mean(np.exp(2j * angle[positive]))))
            reasons[KEYS[2]] = ''
        else:
            reasons[KEYS[2]] = 'no_positive_length_edges'
        log_area = np.log(nodes.area.to_numpy(float))
        a, b = log_area[ij[:, 0]], log_area[ij[:, 1]]
        x, y = np.r_[a, b], np.r_[b, a]
        if np.ptp(x) > 0:
            metrics[KEYS[3]] = float(np.corrcoef(x, y)[0, 1])
            reasons[KEYS[3]] = ''
        else:
            reasons[KEYS[3]] = 'constant_endpoint_log_area'
    metrics.update(graph_nodes=len(nodes), graph_edges=len(ij),
                   graph_components_available=len(t), graph_components_below_floor=int((t.area < area_floor).sum()),
                   graph_components_clipped=int(retained.touches_edge.sum()),
                   graph_zero_length_edges=int((length == 0).sum()),
                   graph_clipped_area=float(retained.loc[retained.touches_edge, 'area'].sum()),
                   graph_node_log_area_median=float(np.log(nodes.area).median()) if len(nodes) else np.nan,
                   graph_node_log_area_iqr=float(np.diff(np.quantile(np.log(nodes.area), [.25, .75]))[0]) if len(nodes) else np.nan)
    for key, reason in reasons.items():
        metrics[key + '_unavailable_reason'] = reason
    return metrics, nodes, edge_table


def usable(table, key, view='bright_usable'):
    """Unknown quality never certifies a mask; grey-pore is a sensitivity view."""
    ok = np.isfinite(pd.to_numeric(table[key], errors='coerce'))
    ok &= table['bright_low_contrast'].eq(False).fillna(False) if 'bright_low_contrast' in table else False
    if view in ('quality_matched', 'ordinary_reference'):
        ok &= table['grey_pore'].eq(False).fillna(False) if 'grey_pore' in table else False
    if view == 'ordinary_reference':
        ref = table.batch.eq('Batch_3')
        ok &= ~ref | table['cracked_known'].eq(False).fillna(False)
    return np.asarray(ok, dtype=bool)
