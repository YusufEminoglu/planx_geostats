# -*- coding: utf-8 -*-
"""How good is a partition?  Quality measures for regions and clusters (v3.12.0).

The plugin could cut a map into regions (SKATER, K-means, and from v3.12
max-p, AZP and REDCAP) but gave no way to judge the result.  These are the
measures esda groups as cluster-quality statistics, written with numpy alone:

* attribute silhouette (Rousseeuw 1987) - is each unit closer to its own
  region than to the nearest other region, in attribute space?
* boundary silhouette (Wolf, Knaap & Rey 2021) - the same, asked only of units
  on a region boundary and only against the regions they touch
* path silhouette (Wolf et al. 2021) - silhouette on the graph's shortest-path
  dissimilarity, so a region must be both similar and reachable
* homogeneity, completeness and V-measure (Rosenberg & Hirschberg 2007) -
  agreement with a reference partition, optionally area-weighted
* R-squared (between-region share of the attribute variance), region-size
  balance (normalised entropy), contiguity and isoperimetric compactness

All take standardised attributes so no field dominates by its units.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def _z(data):
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=0)) / sd


def _distance_rows(z, rows):
    return np.sqrt(((z[rows, None, :] - z[None, :, :]) ** 2).sum(axis=2))


def silhouette_samples(data, labels, chunk=512):
    """Rousseeuw silhouette per unit on Euclidean attribute distance.
    Units in singleton regions get 0, as in scikit-learn."""
    z = _z(data)
    labels = np.asarray(labels)
    n = len(labels)
    uniq = np.unique(labels)
    out = np.zeros(n)
    if len(uniq) < 2:
        return out
    sizes = {k: int(np.sum(labels == k)) for k in uniq}
    for start in range(0, n, chunk):
        rows = np.arange(start, min(n, start + chunk))
        d = _distance_rows(z, rows)
        for r_i, i in enumerate(rows):
            own = labels[i]
            if sizes[own] == 1:
                continue
            a = d[r_i, labels == own].sum() / (sizes[own] - 1)
            b = min(d[r_i, labels == k].mean() for k in uniq if k != own)
            out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def boundary_silhouette(data, labels, adj):
    """Silhouette of boundary units against the regions they actually touch;
    interior units get 0 (esda.boundary_silhouette)."""
    z = _z(data)
    labels = np.asarray(labels)
    out = np.zeros(len(labels))
    for i in range(len(labels)):
        touching = {labels[v] for v in adj[i]} - {labels[i]}
        own = np.flatnonzero(labels == labels[i])
        if not touching or len(own) == 1:
            continue
        d = np.sqrt(((z - z[i]) ** 2).sum(axis=1))
        a = d[own].sum() / (len(own) - 1)
        b = min(d[labels == k].mean() for k in touching)
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def path_silhouette(data, labels, adj, max_units=3000):
    """Silhouette on shortest-path distances through the contiguity graph,
    with each edge weighted by the attribute distance of its two units.
    Returns None above ``max_units`` (all-pairs Dijkstra is O(n E log n))."""
    n = len(labels)
    if n > max_units:
        return None
    z = _z(data)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.zeros(n)
    w = [{v: float(np.sqrt(((z[u] - z[v]) ** 2).sum())) for v in adj[u]} for u in range(n)]
    out = np.zeros(n)
    for s in range(n):
        dist = np.full(n, np.inf)
        dist[s] = 0.0
        heap = [(0.0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, wv in w[u].items():
                nd = d + wv
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        own = labels == labels[s]
        if own.sum() == 1:
            continue
        reach = np.isfinite(dist)
        a = dist[own & reach].sum() / max(1, (own & reach).sum() - 1)
        others = [dist[(labels == k) & reach].mean() for k in uniq if k != labels[s] and np.any((labels == k) & reach)]
        if not others:
            continue
        b = min(others)
        out[s] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def _entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def homogeneity_completeness(labels, reference, weights=None):
    """Rosenberg & Hirschberg (2007) homogeneity, completeness and V-measure.

    ``labels`` is the partition being judged, ``reference`` the one it is
    compared with.  With ``weights`` (unit areas) the contingency table counts
    area instead of units, which is esda's areal version."""
    a = np.asarray(labels)
    b = np.asarray(reference)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    ka, kb = np.unique(a), np.unique(b)
    table = np.array([[w[(a == x) & (b == y)].sum() for y in kb] for x in ka])
    total = table.sum()
    h_ref = _entropy(table.sum(axis=0))
    h_lab = _entropy(table.sum(axis=1))
    # conditional entropies
    h_ref_given_lab = 0.0
    for row in table:
        if row.sum() > 0:
            h_ref_given_lab += row.sum() / total * _entropy(row)
    h_lab_given_ref = 0.0
    for col in table.T:
        if col.sum() > 0:
            h_lab_given_ref += col.sum() / total * _entropy(col)
    homogeneity = 1.0 if h_ref == 0 else 1.0 - h_ref_given_lab / h_ref
    completeness = 1.0 if h_lab == 0 else 1.0 - h_lab_given_ref / h_lab
    v = 0.0 if homogeneity + completeness == 0 else 2 * homogeneity * completeness / (homogeneity + completeness)
    return {"homogeneity": float(homogeneity), "completeness": float(completeness), "v_measure": float(v)}


def partition_summary(data, labels, adj=None, weights=None) -> dict:
    """Region-level and overall quality figures."""
    z = _z(data)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    sst = float((z ** 2).sum())
    ssw = 0.0
    regions = []
    for k in uniq:
        idx = np.flatnonzero(labels == k)
        block = z[idx]
        ssd = float(((block - block.mean(axis=0)) ** 2).sum())
        ssw += ssd
        contiguous = None
        if adj is not None:
            members = set(idx.tolist())
            seen = {idx[0]}
            stack = [idx[0]]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in members and v not in seen:
                        seen.add(v)
                        stack.append(v)
            contiguous = len(seen) == len(idx)
        regions.append({"region": int(k), "size": int(len(idx)), "ssd": ssd,
                        "weight": float(np.sum(weights[idx])) if weights is not None else float(len(idx)),
                        "contiguous": contiguous})
    sizes = [r["size"] for r in regions]
    balance = _entropy(sizes) / math.log(len(sizes)) if len(sizes) > 1 else 1.0
    return {
        "k": int(len(uniq)), "sst": sst, "ssw": ssw,
        "r2": 1.0 - ssw / sst if sst > 0 else float("nan"),
        "balance": float(balance), "regions": regions,
        "all_contiguous": None if adj is None else all(r["contiguous"] for r in regions),
    }


def isoperimetric_quotient(area: float, perimeter: float) -> float:
    """4 pi A / P^2: 1 for a circle, about 0.785 for a square, near 0 for a snake."""
    if perimeter <= 0:
        return float("nan")
    return float(4.0 * math.pi * area / perimeter ** 2)
