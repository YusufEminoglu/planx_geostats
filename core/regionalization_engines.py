# -*- coding: utf-8 -*-
"""Spatially constrained regionalisation: max-p, AZP and REDCAP (v3.12.0).

numpy and the standard library only, like SKATER in advanced_stats_engines -
the plan had these arrive through the optional ``spopt`` package; writing them
here keeps the tools working on a stock QGIS install and lets the smoke suite
test them without QGIS.

Shared conventions
------------------
* ``graph`` is a :class:`core.local_stats_engines.SpatialGraph`; only its
  undirected edge set is used (contiguity), weights are ignored.
* ``data`` is an (n, p) attribute matrix; every method standardises it
  (z-scores, population sd) so no field dominates by its units.
* Heterogeneity is the within-region sum of squared deviations (SSD) of the
  standardised attributes - the objective max-p, AZP and REDCAP all minimise.
* Labels are 0-based region ids; every region is contiguous on the graph.
  Randomised parts take a ``seed`` and are reproducible.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def standardize(data) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def adjacency_lists(graph) -> list[list[int]]:
    adj = [set() for _ in range(graph.n)]
    for a, b in zip(*graph.symmetric_pairs()):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return [sorted(s) for s in adj]


def region_ssd(z: np.ndarray, members) -> float:
    if len(members) < 2:
        return 0.0
    block = z[list(members)]
    return float(((block - block.mean(axis=0)) ** 2).sum())


def total_ssd(z: np.ndarray, labels) -> float:
    labels = np.asarray(labels)
    return float(sum(region_ssd(z, np.flatnonzero(labels == r)) for r in np.unique(labels) if r >= 0))


def is_connected(members, adj) -> bool:
    members = set(members)
    if len(members) <= 1:
        return True
    start = next(iter(members))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in members and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(members)


def components(adj) -> list[list[int]]:
    seen = [False] * len(adj)
    out = []
    for s in range(len(adj)):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        out.append(sorted(comp))
    return out


def relabel(labels) -> np.ndarray:
    labels = np.asarray(labels)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)


class _RegionStats:
    """Running sums per region so a move's SSD change is O(p), not O(n p)."""

    def __init__(self, z, labels):
        self.z = z
        self.labels = np.asarray(labels, dtype=int).copy()
        k = int(self.labels.max()) + 1
        p = z.shape[1]
        self.count = np.zeros(k)
        self.s1 = np.zeros((k, p))
        self.s2 = np.zeros(k)
        for i, r in enumerate(self.labels):
            self._add(i, r)

    def _add(self, i, r, sign=1.0):
        self.count[r] += sign
        self.s1[r] += sign * self.z[i]
        self.s2[r] += sign * float(self.z[i] @ self.z[i])

    def ssd(self, r) -> float:
        c = self.count[r]
        return 0.0 if c <= 0 else float(self.s2[r] - (self.s1[r] @ self.s1[r]) / c)

    def move_delta(self, i, src, dst) -> float:
        zi = self.z[i]
        q = float(zi @ zi)
        c_s, c_d = self.count[src], self.count[dst]
        s_new = (self.s2[src] - q) - ((self.s1[src] - zi) @ (self.s1[src] - zi)) / (c_s - 1) if c_s > 1 else 0.0
        d_new = (self.s2[dst] + q) - ((self.s1[dst] + zi) @ (self.s1[dst] + zi)) / (c_d + 1)
        return float(s_new + d_new - self.ssd(src) - self.ssd(dst))

    def move(self, i, src, dst):
        self._add(i, src, -1.0)
        self._add(i, dst, 1.0)
        self.labels[i] = dst

    def total(self) -> float:
        return float(sum(self.ssd(r) for r in range(len(self.count))))


def _local_search(z, adj, labels, rng, *, floor=None, threshold=None, method="greedy",
                  max_rounds=200, sa_t0=None, sa_cooling=0.95, tabu_length=10, min_size=1):
    """Move border units between regions while keeping every region contiguous.

    ``greedy``  accept only improving moves (AZP basic)
    ``sa``      simulated annealing: worsening moves accepted with exp(-d/T)
    ``tabu``    best admissible move each step, reverse moves tabu for a while
    A donor region must stay connected, non-empty, at least ``min_size`` and -
    for max-p - above the floor threshold.
    """
    stats = _RegionStats(z, labels)
    floor_sum = None
    if floor is not None:
        k = int(stats.labels.max()) + 1
        floor_sum = np.zeros(k)
        for i, r in enumerate(stats.labels):
            floor_sum[r] += floor[i]
    n = len(adj)
    best_labels = stats.labels.copy()
    best_total = stats.total()
    temperature = sa_t0 if sa_t0 is not None else max(best_total / max(n, 1) * 0.1, 1e-9)
    tabu: dict = {}
    step = 0

    def admissible(i, src, dst):
        if stats.count[src] - 1 < max(1, min_size):
            return False
        if floor_sum is not None and floor_sum[src] - floor[i] < threshold:
            return False
        members = np.flatnonzero(stats.labels == src)
        rest = [m for m in members if m != i]
        return is_connected(rest, adj)

    for _round in range(max_rounds):
        improved = False
        if method == "tabu":
            best_move = None
            for i in rng.permutation(n):
                src = stats.labels[i]
                for dst in {stats.labels[v] for v in adj[i]} - {src}:
                    if tabu.get((i, dst), -1) >= step:
                        continue
                    d = stats.move_delta(i, src, dst)
                    if best_move is None or d < best_move[0]:
                        if admissible(i, src, dst):
                            best_move = (d, i, src, dst)
            if best_move is None:
                break
            d, i, src, dst = best_move
            stats.move(i, src, dst)
            if floor_sum is not None:
                floor_sum[src] -= floor[i]
                floor_sum[dst] += floor[i]
            tabu[(i, src)] = step + tabu_length
            step += 1
            total = stats.total()
            if total < best_total - 1e-12:
                best_total, best_labels = total, stats.labels.copy()
                improved = True
            if not improved and step > 3 * n:
                break
            continue
        for i in rng.permutation(n):
            src = stats.labels[i]
            targets = {stats.labels[v] for v in adj[i]} - {src}
            for dst in targets:
                d = stats.move_delta(i, src, dst)
                accept = d < -1e-12
                if not accept and method == "sa" and d > 0:
                    accept = rng.random() < math.exp(-d / temperature)
                if accept and admissible(i, src, dst):
                    stats.move(i, src, dst)
                    if floor_sum is not None:
                        floor_sum[src] -= floor[i]
                        floor_sum[dst] += floor[i]
                    improved = improved or d < -1e-12
                    break
        total = stats.total()
        if total < best_total - 1e-12:
            best_total, best_labels = total, stats.labels.copy()
        if method == "sa":
            temperature *= sa_cooling
            if temperature < 1e-6 and not improved:
                break
        elif not improved:
            break
    return relabel(best_labels), best_total


# ---------------------------------------------------------------------------
# max-p
# ---------------------------------------------------------------------------


def max_p_regions(data, graph, floor, threshold, *, iterations=99, seed=42, local_search="sa",
                  max_rounds=100) -> dict:
    """Max-p-regions (Duque, Anselin & Rey 2012) with the Wei, Rey & Knaap
    (2020) heuristic: as many contiguous regions as possible such that every
    region's sum of ``floor`` is at least ``threshold``; among partitions with
    that maximum number of regions, the least heterogeneous.

    Construction: random-order region growing, repeated ``iterations`` times;
    a growing region takes the adjacent unassigned unit closest to its
    attribute centroid until the floor is met, and units that cannot join a
    region meeting the floor (enclaves) are attached afterwards to the
    adjacent region whose SSD increases least.  Improvement: a local search
    (``sa`` simulated annealing, ``tabu`` or ``greedy``) moves border units
    while keeping regions contiguous and above the floor.
    """
    z = standardize(data)
    f = np.asarray(floor, dtype=float)
    n = len(f)
    if np.any(f < 0):
        raise ValueError("The floor variable must be non-negative.")
    if threshold <= 0:
        raise ValueError("The floor threshold must be positive.")
    adj = adjacency_lists(graph)
    comps = components(adj)
    infeasible = [c for c in comps if f[c].sum() < threshold]
    if len(infeasible) == len(comps):
        raise ValueError("No connected part of the study area reaches the floor threshold; lower it.")
    rng = np.random.default_rng(int(seed))
    best = None
    tried = 0
    for _ in range(max(1, int(iterations))):
        tried += 1
        labels = np.full(n, -1)
        region = 0
        for s in rng.permutation(n):
            if labels[s] != -1:
                continue
            members = [s]
            labels[s] = region
            total = f[s]
            centroid = z[s].copy()
            frontier = set(adj[s])
            while total < threshold:
                cand = [v for v in frontier if labels[v] == -1]
                if not cand:
                    break
                d = [float(((z[v] - centroid) ** 2).sum()) for v in cand]
                v = cand[int(np.argmin(d))]
                labels[v] = region
                members.append(v)
                total += f[v]
                centroid = z[members].mean(axis=0)
                frontier |= set(adj[v])
            if total >= threshold:
                region += 1
            else:
                for m in members:
                    labels[m] = -2          # provisional enclave
        p = region
        if p == 0:
            continue
        labels[labels == -2] = -1
        # attach enclaves; repeat until every reachable enclave is placed
        changed = True
        while changed and np.any(labels == -1):
            changed = False
            for e in np.flatnonzero(labels == -1):
                regions = {labels[v] for v in adj[e] if labels[v] >= 0}
                if not regions:
                    continue
                best_r, best_inc = None, None
                for r in regions:
                    members = np.flatnonzero(labels == r)
                    inc = region_ssd(z, list(members) + [e]) - region_ssd(z, members)
                    if best_inc is None or inc < best_inc:
                        best_r, best_inc = r, inc
                labels[e] = best_r
                changed = True
        ssd = total_ssd(z, labels)
        key = (p, -ssd)
        if best is None or key > best[0]:
            best = (key, labels.copy())
    if best is None:
        raise ValueError("No region could reach the floor threshold; lower it.")
    labels = best[1]
    unassigned = labels < 0                 # units in components that cannot reach the floor
    p = int(labels.max()) + 1
    construction_ssd = total_ssd(z, labels)
    if local_search != "none" and p > 1:
        assigned = np.flatnonzero(~unassigned)
        sub = {old: new for new, old in enumerate(assigned)}
        sub_adj = [[sub[v] for v in adj[u] if v in sub] for u in assigned]
        new_labels, _ = _local_search(z[assigned], sub_adj, labels[assigned], rng, floor=f[assigned],
                                      threshold=threshold, method=local_search, max_rounds=max_rounds)
        labels = labels.copy()
        labels[assigned] = new_labels
    return {
        "labels": labels, "p": int(labels.max()) + 1, "ssd": total_ssd(z, labels),
        "construction_ssd": construction_ssd, "unassigned": int(unassigned.sum()),
        "iterations": tried,
        "floor_by_region": [float(f[labels == r].sum()) for r in range(int(labels.max()) + 1)],
    }


# ---------------------------------------------------------------------------
# AZP
# ---------------------------------------------------------------------------


def _random_contiguous_partition(adj, k, rng):
    """Grow k regions from random seeds, round-robin, so every region is contiguous."""
    n = len(adj)
    labels = np.full(n, -1)
    comps = components(adj)
    if len(comps) > k:
        raise ValueError(f"The neighbourhood graph has {len(comps)} disconnected parts, more than k = {k} regions.")
    # at least one seed per component, the rest spread by component size
    seeds = [int(rng.choice(c)) for c in comps]
    pool = [u for u in range(n) if u not in set(seeds)]
    extra = rng.choice(pool, size=k - len(seeds), replace=False) if k > len(seeds) else []
    seeds += [int(s) for s in extra]
    frontiers = []
    for r, s in enumerate(seeds):
        labels[s] = r
        frontiers.append(set(adj[s]))
    remaining = n - len(seeds)
    while remaining > 0:
        progressed = False
        for r in rng.permutation(len(seeds)):
            cand = [v for v in frontiers[r] if labels[v] == -1]
            if not cand:
                continue
            v = int(rng.choice(cand))
            labels[v] = r
            frontiers[r] |= set(adj[v])
            remaining -= 1
            progressed = True
        if not progressed:
            break
    if remaining:
        raise ValueError("Some units could not be reached from any seed (isolated units). Widen the neighbourhood.")
    return labels


def azp(data, graph, k, *, restarts=10, seed=42, method="greedy", max_rounds=200, min_size=1) -> dict:
    """Automatic Zoning Procedure (Openshaw 1977; Openshaw & Rao 1995).

    Random contiguous starts, each improved by moving border units between
    neighbouring regions while every region stays contiguous; ``method`` is
    ``greedy`` (basic AZP), ``sa`` (AZP-SA) or ``tabu`` (AZP-Tabu).  The best
    of ``restarts`` runs is returned.
    """
    z = standardize(data)
    adj = adjacency_lists(graph)
    n = len(adj)
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of units.")
    rng = np.random.default_rng(int(seed))
    best = None
    start_ssd = []
    for _ in range(max(1, int(restarts))):
        labels = _random_contiguous_partition(adj, k, rng)
        start_ssd.append(total_ssd(z, labels))
        labels, ssd = _local_search(z, adj, labels, rng, method=method, max_rounds=max_rounds, min_size=min_size)
        if best is None or ssd < best[1]:
            best = (labels, ssd)
    return {"labels": best[0], "ssd": best[1], "k": int(best[0].max()) + 1,
            "start_ssd_median": float(np.median(start_ssd)), "restarts": int(restarts)}


# ---------------------------------------------------------------------------
# REDCAP
# ---------------------------------------------------------------------------


REDCAP_LINKAGES = ("single", "average", "complete", "ward")


def redcap_tree(z, adj, linkage="complete"):
    """Stage 1 of REDCAP (Guo 2008): contiguity-constrained full-order clustering.

    Clusters merge only when they are adjacent; the cluster distance is the
    full-order linkage over all unit pairs (Lance-Williams updates), and each
    merge adds to the spanning tree the shortest contiguity edge between the
    two clusters.  Returns the tree's edge list.
    """
    if linkage not in REDCAP_LINKAGES:
        raise ValueError(f"Unknown linkage: {linkage}")
    n = len(adj)
    diff = z[:, None, :] - z[None, :, :]
    d2 = (diff ** 2).sum(axis=2)
    dist = d2.copy() if linkage == "ward" else np.sqrt(d2)
    if linkage == "ward":
        dist = dist / 2.0                      # Ward merge cost of two singletons
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    nbrs = [set(a) for a in adj]
    # connecting edges between clusters: best (attribute) edge for each adjacent pair
    link = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                link[(u, v)] = (float(np.sqrt(d2[u, v])), u, v)
    heap = [(dist[u, v], u, v) for u in range(n) for v in adj[u] if u < v]
    heapq.heapify(heap)
    tree = []
    while heap and len(tree) < n - 1:
        dd, a, b = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or b not in nbrs[a] or abs(dd - dist[a, b]) > 1e-12:
            continue
        edge = link[(min(a, b), max(a, b))]
        tree.append((edge[1], edge[2]))
        # merge b into a
        na, nb = size[a], size[b]
        others = np.flatnonzero(alive)
        others = others[(others != a) & (others != b)]
        da, db = dist[a, others], dist[b, others]
        if linkage == "single":
            new = np.minimum(da, db)
        elif linkage == "complete":
            new = np.maximum(da, db)
        elif linkage == "average":
            new = (na * da + nb * db) / (na + nb)
        else:
            nc = size[others]
            new = ((na + nc) * da + (nb + nc) * db - nc * dist[a, b]) / (na + nb + nc)
        dist[a, others] = new
        dist[others, a] = new
        size[a] = na + nb
        alive[b] = False
        merged = (nbrs[a] | nbrs[b]) - {a, b}
        for c in nbrs[b]:
            nbrs[c].discard(b)
            if c != a:
                nbrs[c].add(a)
        nbrs[a] = merged
        for c in merged:
            key_ab = (min(a, c), max(a, c))
            key_bc = (min(b, c), max(b, c))
            cand = [link[k] for k in (key_ab, key_bc) if k in link]
            link[key_ab] = min(cand)
            link.pop(key_bc, None)
            heapq.heappush(heap, (dist[a, c], min(a, c), max(a, c)))
    return tree


def _cut_tree(z, n, tree, k, min_size=1):
    """Stage 2: remove k-1 tree edges, each time the one that most reduces SSD."""
    tadj = [[] for _ in range(n)]
    for u, v in tree:
        tadj[u].append(v)
        tadj[v].append(u)
    labels = np.zeros(n, dtype=int)
    removed = set()
    for region_count in range(1, k):
        best = None
        for r in range(region_count):
            members = np.flatnonzero(labels == r)
            if len(members) < 2 * max(1, min_size):
                continue
            root = int(members[0])
            parent = {root: -1}
            order = [root]
            for u in order:
                for v in tadj[u]:
                    if (min(u, v), max(u, v)) in removed or v in parent:
                        continue
                    parent[v] = u
                    order.append(v)
            cnt = {u: 1.0 for u in order}
            s1 = {u: z[u].copy() for u in order}
            s2 = {u: float(z[u] @ z[u]) for u in order}
            for u in reversed(order[1:]):
                p = parent[u]
                cnt[p] += cnt[u]
                s1[p] = s1[p] + s1[u]
                s2[p] += s2[u]
            tot_c, tot_s1, tot_s2 = cnt[root], s1[root], s2[root]
            base = tot_s2 - tot_s1 @ tot_s1 / tot_c
            for u in order[1:]:
                c_in, c_out = cnt[u], tot_c - cnt[u]
                if c_in < min_size or c_out < min_size:
                    continue
                ssd_in = s2[u] - s1[u] @ s1[u] / c_in
                out_s1 = tot_s1 - s1[u]
                ssd_out = (tot_s2 - s2[u]) - out_s1 @ out_s1 / c_out
                gain = base - ssd_in - ssd_out
                if best is None or gain > best[0]:
                    best = (gain, r, u, parent[u])
        if best is None:
            raise ValueError(f"Cannot cut the tree into {k} regions of at least {min_size} units.")
        _gain, r, u, p = best
        removed.add((min(u, p), max(u, p)))
        # relabel the subtree under u
        stack, seen = [u], {u}
        while stack:
            a = stack.pop()
            labels[a] = region_count
            for b in tadj[a]:
                if b not in seen and (min(a, b), max(a, b)) not in removed and labels[b] == r:
                    seen.add(b)
                    stack.append(b)
    return labels


def redcap(data, graph, k, *, linkage="complete", min_size=1) -> dict:
    """REDCAP (Guo 2008): build a spanning tree by contiguity-constrained
    full-order clustering, then cut it into k regions by the SKATER-style
    greatest-SSD-reduction rule.  Deterministic.  Full-order complete linkage
    is Guo's best performer and the default.
    """
    z = standardize(data)
    adj = adjacency_lists(graph)
    n = len(adj)
    comps = components(adj)
    if len(comps) > 1:
        raise ValueError(f"The neighbourhood graph has {len(comps)} disconnected parts; REDCAP needs one. Widen the neighbourhood.")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and the number of units.")
    if n > 4000:
        raise ValueError("REDCAP keeps an n x n distance matrix; above 4000 units use Max-p or AZP.")
    tree = redcap_tree(z, adj, linkage)
    labels = _cut_tree(z, n, tree, k, min_size)
    return {"labels": labels, "ssd": total_ssd(z, labels), "k": int(labels.max()) + 1, "tree_edges": len(tree)}
