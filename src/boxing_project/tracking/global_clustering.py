from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from .matcher import cosine_similarity
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


def _build_cluster_prototypes(embs: np.ndarray, gids: np.ndarray) -> dict[int, np.ndarray]:
    """
    Build normalized prototype embedding for each global id found in gids.

    Example:
        gids may contain {1, 2, 3}
    """
    prototypes: dict[int, np.ndarray] = {}
    for gid in sorted(set(int(x) for x in gids.tolist())):
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
    initial_gids: np.ndarray,
) -> dict[NodeKey, int]:
    """
    Refine assignments inside each epoch.

    Rules:
    - global ids 1 and 2 are the two boxer identities
    - inside one epoch, at most one track may be assigned to 1
    - inside one epoch, at most one track may be assigned to 2
    - all remaining tracks get global id 3

    The initial spectral assignment is used only to decide which remaining tracks
    should stay as 3.
    """
    by_epoch: dict[int, list[int]] = {}
    for i, (epoch_id, _local_id) in enumerate(nodes):
        by_epoch.setdefault(int(epoch_id), []).append(i)

    mapping: dict[NodeKey, int] = {}

    for epoch_id in sorted(by_epoch.keys()):
        idxs = sorted(by_epoch[epoch_id], key=lambda i: int(nodes[i][1]))
        if not idxs:
            continue

        sims = {
            i: {
                1: float(cosine_similarity(embs[i], prototypes[1])),
                2: float(cosine_similarity(embs[i], prototypes[2])),
            }
            for i in idxs
        }

        # Pick best candidate for boxer 1
        best_for_1 = max(idxs, key=lambda i: sims[i][1])
        # Pick best candidate for boxer 2 among remaining tracks
        remaining_for_2 = [i for i in idxs if i != best_for_1]

        if remaining_for_2:
            best_for_2 = max(remaining_for_2, key=lambda i: sims[i][2])

            score_ab = sims[best_for_1][1] + sims[best_for_2][2]
            score_ba = sims[best_for_1][2] + sims[best_for_2][1]

            # If swapped assignment is better, swap the two chosen tracks.
            if score_ba > score_ab:
                best_for_1, best_for_2 = best_for_2, best_for_1
        else:
            best_for_2 = None

        # Assign boxer IDs 1 and 2
        mapping[nodes[best_for_1]] = 1
        if best_for_2 is not None:
            mapping[nodes[best_for_2]] = 2

        # All remaining tracks become global id 3
        for i in idxs:
            if i == best_for_1 or i == best_for_2:
                continue
            mapping[nodes[i]] = 3

    return mapping


@dataclass
class GlobalTrackClusterer:
    """
    Offline global ID unification based on:
    1) custom mutual-kNN graph construction
    2) Spectral Clustering over the graph affinity matrix

    Default intended setup:
    - 3 clusters
    - global id 1 = boxer 1
    - global id 2 = boxer 2
    - global id 3 = other people / noise
    """

    k: int = 5
    sim_threshold: float = 0.5
    n_clusters: int = 3
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

        # Allow 3 clusters (or fewer if too few nodes exist).
        n_clusters = max(2, min(int(self.n_clusters), n_nodes))

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=int(self.random_state),
            assign_labels=str(self.assign_labels),
        )
        labels = model.fit_predict(affinity).astype(np.int32)

        # Deterministic remap of spectral labels to global IDs {1, 2, 3, ...}
        unique_labels = sorted(set(int(lbl) for lbl in labels))
        label_to_gid = {lbl: gid for gid, lbl in enumerate(unique_labels, start=1)}
        gids = np.asarray([label_to_gid[int(lbl)] for lbl in labels], dtype=np.int32)

        # Build prototypes for all current global ids.
        prototypes = _build_cluster_prototypes(embs=embs, gids=gids)

        # Make sure ids 1 and 2 always exist for refinement.
        if 1 not in prototypes and 2 in prototypes:
            prototypes[1] = prototypes[2]
        if 2 not in prototypes and 1 in prototypes:
            prototypes[2] = prototypes[1]

        # If cluster 3 does not exist because there are too few nodes / clusters,
        # refinement still works for ids 1 and 2.
        return _refine_epoch_assignments(
            nodes=nodes,
            embs=embs,
            prototypes=prototypes,
            initial_gids=gids,
        )