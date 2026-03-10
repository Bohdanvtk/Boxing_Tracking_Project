from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import SpectralClustering

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


def _adjacency_to_affinity(adj: list[dict[int, float]]) -> np.ndarray:
    """Convert weighted adjacency list to a dense symmetric affinity matrix."""
    n_nodes = len(adj)
    affinity = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, neigh in enumerate(adj):
        for j, w in neigh.items():
            affinity[i, int(j)] = float(w)

    affinity = 0.5 * (affinity + affinity.T)
    # Diagonal self-affinity improves stability for sparse/disconnected graphs.
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized vector (safe for near-zero norms)."""
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _build_boxer_prototypes(embs: np.ndarray, gids: np.ndarray) -> dict[int, np.ndarray]:
    """Build normalized prototype embedding for each boxer gid in {1, 2}."""
    prototypes: dict[int, np.ndarray] = {}
    for gid in (1, 2):
        idxs = np.where(gids == gid)[0]
        if idxs.size == 0:
            continue
        mean_emb = embs[idxs].mean(axis=0)
        prototypes[gid] = _l2_normalize(mean_emb)
    return prototypes


def _refine_epoch_assignments(
    nodes: list[NodeKey],
    embs: np.ndarray,
    prototypes: dict[int, np.ndarray],
) -> dict[NodeKey, int]:
    """
    Enforce one-to-one boxer assignment inside each epoch.

    After global 2-boxer grouping, each epoch is refined so each boxer id (1/2)
    appears at most once in that epoch.
    """
    by_epoch: dict[int, list[int]] = {}
    for i, (epoch_id, _local_id) in enumerate(nodes):
        by_epoch.setdefault(int(epoch_id), []).append(i)

    mapping: dict[NodeKey, int] = {}

    for epoch_id in sorted(by_epoch.keys()):
        idxs = sorted(by_epoch[epoch_id], key=lambda i: int(nodes[i][1]))
        if not idxs:
            continue

        sims = {i: {gid: float(np.dot(embs[i], prototypes[gid])) for gid in (1, 2)} for i in idxs}

        if len(idxs) == 1:
            i = idxs[0]
            gid = 1 if sims[i][1] >= sims[i][2] else 2
            mapping[nodes[i]] = gid
            continue

        if len(idxs) == 2:
            a, b = idxs
            score_ab = sims[a][1] + sims[b][2]
            score_ba = sims[a][2] + sims[b][1]
            if score_ab > score_ba:
                mapping[nodes[a]] = 1
                mapping[nodes[b]] = 2
            elif score_ba > score_ab:
                mapping[nodes[a]] = 2
                mapping[nodes[b]] = 1
            else:
                # deterministic tie-break by smallest local id assigned to boxer 1
                if int(nodes[a][1]) <= int(nodes[b][1]):
                    mapping[nodes[a]] = 1
                    mapping[nodes[b]] = 2
                else:
                    mapping[nodes[a]] = 2
                    mapping[nodes[b]] = 1
            continue

        # len(idxs) > 2: best one-to-one assignment with at most one track per boxer.
        candidates: list[tuple[float, dict[int, int]]] = []
        candidates.append((0.0, {}))

        for i in idxs:
            candidates.append((sims[i][1], {1: i}))
            candidates.append((sims[i][2], {2: i}))

        for i in idxs:
            for j in idxs:
                if i == j:
                    continue
                candidates.append((sims[i][1] + sims[j][2], {1: i, 2: j}))

        best_score, best_assign = candidates[0]

        def tie_key(assign: dict[int, int]) -> tuple:
            # Prefer more assigned boxers; then smallest local ids deterministically.
            local1 = int(nodes[assign[1]][1]) if 1 in assign else 10**9
            local2 = int(nodes[assign[2]][1]) if 2 in assign else 10**9
            return (len(assign), -local1, -local2)

        for score, assign in candidates[1:]:
            if score > best_score:
                best_score, best_assign = score, assign
            elif score == best_score and tie_key(assign) > tie_key(best_assign):
                best_score, best_assign = score, assign

        for gid, i in sorted(best_assign.items()):
            mapping[nodes[i]] = int(gid)

    return mapping


@dataclass
class GlobalTrackClusterer:
    """
    Offline global ID unification based on:
    1) custom mutual-kNN graph construction
    2) Spectral Clustering over the graph affinity matrix.

    The pipeline enforces exactly 2 boxer clusters when there are at least 2 nodes.
    """

    k: int = 5
    sim_threshold: float = 0.5
    n_clusters: int = 2
    random_state: int = 42
    assign_labels: str = "kmeans"

    def build_mapping(self, epoch_tracks: dict[int, dict[int, "Track"]]) -> dict[NodeKey, int]:
        """Build ``(epoch_id, local_track_id) -> global_track_id`` mapping."""
        nodes, embs = build_mean_embeddings(epoch_tracks)
        n_nodes = len(nodes)
        if n_nodes == 0:
            return {}
        if n_nodes == 1:
            return {nodes[0]: 1}

        # Keep custom graph logic (mutual-kNN, threshold, same-epoch suppression).
        adj = build_mutual_knn_graph(nodes, embs, k=self.k, sim_threshold=self.sim_threshold)
        affinity = _adjacency_to_affinity(adj)

        # Spectral clustering is the global grouping stage (exactly 2 boxers).
        n_clusters = 2 if int(self.n_clusters) != 2 else int(self.n_clusters)
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=int(self.random_state),
            assign_labels=str(self.assign_labels),
        )
        labels = model.fit_predict(affinity)

        # Deterministic remap to boxer IDs {1, 2}.
        unique_labels = sorted(set(int(lbl) for lbl in labels))
        if len(unique_labels) == 1:
            labels = labels.copy()
            labels[1] = int(labels[0]) + 1
            unique_labels = sorted(set(int(lbl) for lbl in labels))
        label_to_gid = {lbl: gid for gid, lbl in enumerate(unique_labels, start=1)}
        gids = np.asarray([label_to_gid[int(lbl)] for lbl in labels], dtype=np.int32)

        # Build boxer prototypes from global clusters and refine per-epoch one-to-one assignment.
        prototypes = _build_boxer_prototypes(embs=embs, gids=gids)
        if 1 not in prototypes and 2 in prototypes:
            prototypes[1] = prototypes[2]
        if 2 not in prototypes and 1 in prototypes:
            prototypes[2] = prototypes[1]

        return _refine_epoch_assignments(nodes=nodes, embs=embs, prototypes=prototypes)
