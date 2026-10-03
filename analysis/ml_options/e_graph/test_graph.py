import numpy as np
import pandas as pd
import pytest

from analysis.ml_options.e_graph.graph import KEYS, build_graph, usable


def table(n=12):
    return pd.DataFrame({'label': np.arange(n)+1, 'area': 50+np.arange(n)*10,
                         'centroid-0': np.zeros(n)+10, 'centroid-1': np.arange(n)*20+10,
                         'touches_edge': False})


def test_knn_union_is_unique_symmetric_without_self_edges():
    m, nodes, edges = build_graph(table())
    assert (edges.node_a < edges.node_b).all()
    assert not edges[['node_a', 'node_b']].duplicated().any()
    degree = np.bincount(np.r_[edges.node_a, edges.node_b], minlength=len(nodes))
    assert degree.min() >= 3
    assert m[KEYS[2]] == pytest.approx(1)


def test_row_order_and_equal_distance_ties_are_deterministic():
    t = table()
    t['centroid-0'] = np.repeat([10, 30, 50], 4)
    t['centroid-1'] = np.tile([10, 30, 50, 70], 3)
    a = build_graph(t)
    b = build_graph(t.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(a[1], b[1])
    pd.testing.assert_frame_equal(a[2], b[2])
    for k in KEYS: assert a[0][k] == pytest.approx(b[0][k])


def test_clipped_and_small_nodes_are_excluded_before_topology():
    t = table(14)
    t.loc[0, 'area'] = 49
    t.loc[1, 'touches_edge'] = True
    m, nodes, _ = build_graph(t)
    assert len(nodes) == 12 and m['graph_components_below_floor'] == 1
    assert m['graph_components_clipped'] == 1
    assert not set([1, 2]) & set(nodes.label)


def test_insufficient_geometry_and_constant_area_abstain():
    m, _, _ = build_graph(table(9))
    assert all(np.isnan(m[k]) for k in KEYS)
    t = table(); t['area'] = 100
    m, _, _ = build_graph(t)
    assert np.isnan(m[KEYS[3]]) and m[KEYS[3]+'_unavailable_reason'] == 'constant_endpoint_log_area'


def test_assortativity_matches_both_directed_copies():
    m, nodes, edges = build_graph(table())
    a = np.log(nodes.area.to_numpy())[edges.node_a]
    b = np.log(nodes.area.to_numpy())[edges.node_b]
    expected = np.corrcoef(np.r_[a, b], np.r_[b, a])[0, 1]
    assert m[KEYS[3]] == pytest.approx(expected)


def test_scale_and_rotation_behavior():
    t = table(); a = build_graph(t)[0]
    t[['centroid-0', 'centroid-1']] *= 2
    b = build_graph(t)[0]
    assert b[KEYS[0]] == pytest.approx(a[KEYS[0]]*2)
    for k in KEYS[1:]: assert b[k] == pytest.approx(a[k])
    t[['centroid-0', 'centroid-1']] = t[['centroid-1', 'centroid-0']].to_numpy()
    assert build_graph(t)[0][KEYS[2]] == pytest.approx(a[KEYS[2]])


def test_coincident_centroids_do_not_create_self_loops():
    t = table(); t[['centroid-0', 'centroid-1']] = 10
    m, _, e = build_graph(t)
    assert (e.node_a < e.node_b).all() and m[KEYS[0]] == 0
    assert np.isnan(m[KEYS[1]]) and np.isnan(m[KEYS[2]])


def test_missing_quality_abstains_and_reference_only_is_ordinary():
    t = pd.DataFrame({'batch': ['Batch_1', 'Batch_3', 'Batch_3'], KEYS[0]: [1., 2., 3.],
                      'bright_low_contrast': [False, None, False], 'grey_pore': False,
                      'cracked_known': [True, False, True]})
    assert usable(t, KEYS[0]).tolist() == [True, False, True]
    assert usable(t, KEYS[0], 'ordinary_reference').tolist() == [True, False, False]
    assert not usable(t.drop(columns='bright_low_contrast'), KEYS[0]).any()


@pytest.mark.parametrize('column,value', [('area', 0), ('touches_edge', None), ('centroid-0', np.nan)])
def test_invalid_input_is_explicit(column, value):
    t = table(); t.loc[0, column] = value
    with pytest.raises(ValueError): build_graph(t)
