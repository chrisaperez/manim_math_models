"""GNN message passing & the depth barrier (CS246 HW4.1), pure numpy-free python.

Sum aggregation:  h_v^{k+1} = h_v^k + sum_{u in N(v)} h_u^k,  h_v^0 = 1.

The two demo graphs reproduce Chris's pset trace exactly:
  red node:      1, 2, 6, 18, then 56 (cycle graph) vs 55 (branched path)
  its neighbor:  1, 4, 12, then 38 vs 37
Four layers of message passing are needed to tell the red nodes apart, because
h_v^k is a function of v's k-hop neighborhood only.
"""

from __future__ import annotations


def demo_graphs() -> tuple[dict, dict, str]:
    """(cycle_graph, branched_path_graph, red_node_name).

    Cycle graph: red pendant on a 4-cycle    Branched path: red pendant on a Y
        r - a               r - a - b1 - c1 - d
        a - b1, a - b2      a - b2 - c2
        b1 - c, b2 - c
    Both give the red node's neighbor degree 3 and identical 3-hop trees.
    """
    cycle = {
        "r": ["a"],
        "a": ["r", "b1", "b2"],
        "b1": ["a", "c"],
        "b2": ["a", "c"],
        "c": ["b1", "b2"],
    }
    branched = {
        "r": ["a"],
        "a": ["r", "b1", "b2"],
        "b1": ["a", "c1"],
        "b2": ["a", "c2"],
        "c1": ["b1", "d"],
        "c2": ["b2"],
        "d": ["c1"],
    }
    return cycle, branched, "r"


def run_layers(adj: dict, k: int) -> dict:
    """{node: [h^0, h^1, ..., h^k]} under sum aggregation from all-ones."""
    h = {v: 1 for v in adj}
    hist = {v: [1] for v in adj}
    for _ in range(k):
        h = {v: h[v] + sum(h[u] for u in adj[v]) for v in adj}
        for v in adj:
            hist[v].append(h[v])
    return hist


def divergence_layer(adj_a: dict, adj_b: dict, va: str, vb: str,
                     kmax: int = 12) -> int | None:
    """First layer k at which the two target embeddings differ."""
    ha = run_layers(adj_a, kmax)[va]
    hb = run_layers(adj_b, kmax)[vb]
    for k in range(kmax + 1):
        if ha[k] != hb[k]:
            return k
    return None


def hop_distances(adj: dict, src: str) -> dict:
    """BFS distances from src."""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for v in frontier:
            for u in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    nxt.append(u)
        frontier = nxt
    return dist


def bfs_via_max_aggregation(adj: dict, seed: str, steps: int) -> list[dict]:
    """HW4 1(c): a GNN layer with max-aggregation + OR simulates one BFS
    frontier step per layer. Returns the binary state after each layer."""
    state = {v: 1 if v == seed else 0 for v in adj}
    out = [dict(state)]
    for _ in range(steps):
        state = {v: max([state[v]] + [state[u] for u in adj[v]]) for v in adj}
        out.append(dict(state))
    return out
