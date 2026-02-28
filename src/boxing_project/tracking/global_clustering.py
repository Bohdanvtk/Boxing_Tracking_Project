from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .track import Track


NodeKey = tuple[int, int]


def _track_mean_embedding(track: "Track") -> np.ndarray | None:
    """Compute an L2-normalized mean embedding from ``track.app_emb_history``."""
    emb_history = list(getattr(track, "app_emb_history", []) or [])
    if not emb_history:
        return None

    embs = np.asarray(emb_history, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return None

    mean_emb = embs.mean(axis=0)
    if not np.all(np.isfinite(mean_emb)):
        return None

    norm = float(np.linalg.norm(mean_emb))
    if norm <= 1e-8:
        return None

    return (mean_emb / norm).astype(np.float32)


def build_mean_embeddings(
    epoch_tracks: dict[int, dict[int, "Track"]],
) -> tuple[list[NodeKey], np.ndarray]:
    """
    Build track-level mean embeddings.

    Returns a list of nodes ``(epoch_id, local_track_id)`` and a float32 matrix of
    corresponding L2-normalized embeddings with shape ``(N, D)``.
    Tracks without a valid mean embedding are skipped.
    """
    nodes: list[NodeKey] = []
    embs: list[np.ndarray] = []

    for epoch_id in sorted(epoch_tracks.keys()):
        tracks_by_id = epoch_tracks.get(epoch_id, {})
        for local_id, track in sorted(tracks_by_id.items()):
            mean_emb = _track_mean_embedding(track)
            if mean_emb is None:
                continue
            nodes.append((int(epoch_id), int(local_id)))
            embs.append(mean_emb)

    if not embs:
        return nodes, np.zeros((0, 0), dtype=np.float32)

    emb_matrix = np.vstack(embs).astype(np.float32)
    return nodes, emb_matrix


def build_mutual_knn_graph(
    nodes: list[NodeKey],
    embs: np.ndarray,
    k: int,
    sim_threshold: float,
) -> list[dict[int, float]]:
    """
    Build a weighted mutual-kNN graph using cosine similarity.

    ``adj[i]`` is a dictionary ``{j: sim_ij}``.
    Edge ``i--j`` exists only if:
    1) ``j`` is in top-k neighbors of ``i`` and ``i`` is in top-k of ``j`` (mutual kNN)
    2) similarity is at least ``sim_threshold``
    3) nodes belong to different epochs (no intra-epoch edges).
    """
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2:
        raise ValueError("embs must be a 2D array")

    n_nodes = int(embs.shape[0])
    if len(nodes) != n_nodes:
        raise ValueError("nodes length must match embs.shape[0]")
    if n_nodes == 0:
        return []

    k_eff = max(0, min(int(k), n_nodes - 1))
    adj: list[dict[int, float]] = [dict() for _ in range(n_nodes)]
    if k_eff == 0:
        return adj

    sim = embs @ embs.T
    np.fill_diagonal(sim, -np.inf)

    topk_sets: list[set[int]] = []
    for i in range(n_nodes):
        row = sim[i]
        idx = np.argpartition(row, -k_eff)[-k_eff:]
        valid = {int(j) for j in idx if np.isfinite(row[j]) and float(row[j]) >= float(sim_threshold)}
        topk_sets.append(valid)

    for i in range(n_nodes):
        for j in topk_sets[i]:
            if i >= j:
                continue
            if i in topk_sets[j]:
                # Skip edges between tracks from the same epoch.
                if nodes[i][0] == nodes[j][0]:
                    continue
                w = float(sim[i, j])
                adj[i][j] = w
                adj[j][i] = w

    return adj


def chinese_whispers(
    adj: list[dict[int, float]],
    num_iters: int,
    rng_seed: int,
) -> list[int]:
    """
    Run Chinese Whispers on a weighted graph.

    Initialization: ``labels[i] = i``.
    Update rule per node: pick neighbor-label with maximum summed edge weight.
    Tie-breaking is deterministic: higher score first, then smaller label id.
    """
    n_nodes = len(adj)
    labels = list(range(n_nodes))
    if n_nodes == 0:
        return labels

    rng = np.random.default_rng(int(rng_seed))
    order = np.arange(n_nodes, dtype=np.int32)

    for _ in range(max(0, int(num_iters))):
        rng.shuffle(order)
        for idx in order:
            i = int(idx)
            neigh = adj[i]
            if not neigh:
                continue

            scores: dict[int, float] = {}
            for j, w in neigh.items():
                lbl = labels[int(j)]
                scores[lbl] = scores.get(lbl, 0.0) + float(w)

            best_label = min(scores.keys())
            best_score = scores[best_label]
            for lbl, score in scores.items():
                if (score > best_score) or (score == best_score and lbl < best_label):
                    best_label = lbl
                    best_score = score
            labels[i] = best_label

    return labels


def _support_score(i: int, nodes: list[NodeKey], labels: list[int], adj: list[dict[int, float]]) -> float:
    """
    Compute cross-epoch support for node ``i`` within its current label.

    score = sum of edge weights to neighbors that:
      - have the same label as node ``i``
      - belong to a different epoch.
    """
    epoch_i = int(nodes[i][0])
    label_i = int(labels[i])
    score = 0.0
    for j, w in adj[i].items():
        if int(labels[int(j)]) != label_i:
            continue
        if int(nodes[int(j)][0]) == epoch_i:
            continue
        score += float(w)
    return float(score)


def _enforce_unique_gid_per_epoch(
    nodes: list[NodeKey],
    labels: list[int],
    label_to_gid: dict[int, int],
    adj: list[dict[int, float]],
) -> dict[NodeKey, int]:
    """Ensure that one epoch cannot map multiple local tracks to the same global ID."""
    mapping: dict[NodeKey, int] = {}
    for i, node in enumerate(nodes):
        mapping[node] = int(label_to_gid[int(labels[i])])

    groups: dict[tuple[int, int], list[int]] = {}
    for i, node in enumerate(nodes):
        gid = int(mapping[node])
        key = (int(node[0]), gid)
        groups.setdefault(key, []).append(i)

    next_gid = (max(label_to_gid.values()) + 1) if label_to_gid else 1

    for (_, gid), idxs in sorted(groups.items()):
        if len(idxs) <= 1:
            continue

        # Keep node with strongest cross-epoch support; tie-break by smallest local_id.
        keep_idx = min(
            idxs,
            key=lambda i: (-_support_score(i, nodes, labels, adj), int(nodes[i][1]), i),
        )

        for i in sorted(idxs):
            if i == keep_idx:
                continue
            mapping[nodes[i]] = int(next_gid)
            next_gid += 1

    return mapping


@dataclass
class GlobalTrackClusterer:
    """Offline global ID unification based on mutual-kNN + Chinese Whispers."""

    k: int = 10
    sim_threshold: float = 0.5
    num_iters: int = 20
    rng_seed: int = 0

    def build_mapping(self, epoch_tracks: dict[int, dict[int, "Track"]]) -> dict[NodeKey, int]:
        """Build ``(epoch_id, local_track_id) -> global_track_id`` mapping."""
        nodes, embs = build_mean_embeddings(epoch_tracks)
        n_nodes = len(nodes)
        if n_nodes == 0:
            return {}

        adj = build_mutual_knn_graph(nodes, embs, k=self.k, sim_threshold=self.sim_threshold)
        labels = chinese_whispers(adj, num_iters=self.num_iters, rng_seed=self.rng_seed)

        unique_labels = sorted(set(int(lbl) for lbl in labels))
        label_to_gid = {lbl: gid for gid, lbl in enumerate(unique_labels, start=1)}

        return _enforce_unique_gid_per_epoch(
            nodes=nodes,
            labels=labels,
            label_to_gid=label_to_gid,
            adj=adj,
        )

