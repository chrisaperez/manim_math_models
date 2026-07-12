"""Demo 4's math must reproduce Chris's HW4 trace digit for digit."""

from core.sims import gnn


CYCLE, BRANCHED, RED = gnn.demo_graphs()


def test_red_traces_match_the_pset():
    assert gnn.run_layers(CYCLE, 4)[RED] == [1, 2, 6, 18, 56]
    assert gnn.run_layers(BRANCHED, 4)[RED] == [1, 2, 6, 18, 55]


def test_neighbor_traces_match_the_pset():
    assert gnn.run_layers(CYCLE, 3)["a"] == [1, 4, 12, 38]
    assert gnn.run_layers(BRANCHED, 3)["a"] == [1, 4, 12, 37]


def test_divergence_at_exactly_four_layers():
    assert gnn.divergence_layer(CYCLE, BRANCHED, RED, RED) == 4


def test_khop_views_identical_before_divergence():
    """Every node at distance <= d from red has matching (k-d)-hop info; the
    concrete check: values at layer k agree for all k < 4."""
    ha = gnn.run_layers(CYCLE, 3)[RED]
    hb = gnn.run_layers(BRANCHED, 3)[RED]
    assert ha == hb


def test_bfs_simulation_matches_hop_distances():
    dist = gnn.hop_distances(BRANCHED, "r")
    states = gnn.bfs_via_max_aggregation(BRANCHED, "r", steps=5)
    for k, state in enumerate(states):
        for v, on in state.items():
            assert on == (1 if dist[v] <= k else 0)
