from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .track import Track


# A local track is identified by two numbers:
#   epoch_id       - fragment/shot id after reset boundaries
#   local_track_id - id produced by the local tracker inside that epoch
NodeKey = tuple[int, int]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _mean_l2_feature(history: list[np.ndarray]) -> np.ndarray | None:
    """Average a track's appearance history and return one normalized feature.

    Returns None when the history is empty, malformed, non-finite, or degenerate.
    Keeping this strict is useful: global clustering should use only reliable
    appearance vectors.
    """
    if not history:
        return None

    feats = np.asarray(history, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return None
    if not np.all(np.isfinite(feats)):
        return None

    mean_feat = feats.mean(axis=0)
    if not np.all(np.isfinite(mean_feat)):
        return None

    norm = float(np.linalg.norm(mean_feat))
    if norm <= 1e-8:
        return None

    return (mean_feat / norm).astype(np.float32)


def _track_mean_embedding(track: "Track") -> np.ndarray | None:
    """Return the track-level appearance descriptor used by global grouping."""
    # app_emb_history already stores fused e_app descriptors, so global grouping
    # only averages them into one track-level descriptor.
    return _mean_l2_feature(list(getattr(track, "app_emb_history", []) or []))


def build_mean_embeddings(
    epoch_tracks: dict[int, dict[int, "Track"]],
    *,
    confirmed_only: bool = True,
    min_track_history: int = 5,
) -> tuple[list[NodeKey], np.ndarray]:
    """Collect valid local tracks and their mean appearance embeddings.

    Output:
        nodes:
            [(epoch_id, local_track_id), ...]
        embeddings:
            float32 matrix of shape (N, D), already L2-normalized per row.

    Tracks are skipped when:
        - confirmed_only=True and track.confirmed is false
        - app_emb_history is shorter than min_track_history
        - the mean feature cannot be computed safely
    """
    nodes: list[NodeKey] = []
    embs: list[np.ndarray] = []
    min_history = max(0, int(min_track_history))

    for epoch_id in sorted(epoch_tracks.keys()):
        tracks_by_id = epoch_tracks.get(epoch_id, {}) or {}
        for local_id, track in sorted(tracks_by_id.items()):
            if confirmed_only and not bool(getattr(track, "confirmed", False)):
                continue

            history = list(getattr(track, "app_emb_history", []) or [])
            if len(history) < min_history:
                continue

            mean_emb = _track_mean_embedding(track)
            if mean_emb is None:
                continue

            nodes.append((int(epoch_id), int(local_id)))
            embs.append(mean_emb)

    if not embs:
        return nodes, np.zeros((0, 0), dtype=np.float32)

    return nodes, np.vstack(embs).astype(np.float32)


def _cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """Build a safe cosine-similarity matrix for track-level embeddings."""
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized = np.divide(
        embs,
        np.maximum(norms, 1e-8),
        out=np.zeros_like(embs),
        where=np.isfinite(norms),
    )

    sim = normalized @ normalized.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


# ---------------------------------------------------------------------------
# Group quality helpers
# ---------------------------------------------------------------------------

def _pairwise_values(group: set[int] | list[int], sim: np.ndarray) -> list[float]:
    """Return all pairwise similarities inside a group."""
    idxs = sorted(int(i) for i in group)
    return [float(sim[i, j]) for i, j in combinations(idxs, 2)]


def _group_mean_similarity(group: set[int] | list[int], sim: np.ndarray) -> float:
    """Mean internal similarity. Singletons are treated as perfectly self-consistent."""
    values = _pairwise_values(group, sim)
    return 1.0 if not values else float(np.mean(values))


def _group_min_similarity(group: set[int] | list[int], sim: np.ndarray) -> float:
    """Worst internal similarity. This is important for strict-clique debugging."""
    values = _pairwise_values(group, sim)
    return 1.0 if not values else float(np.min(values))


def _has_epoch_conflict(group: set[int] | list[int], nodes: list[NodeKey]) -> bool:
    """True if one group would contain two local tracks from the same epoch."""
    epochs = [int(nodes[i][0]) for i in group]
    return len(epochs) != len(set(epochs))


def _is_strict_clique(group: set[int] | list[int], sim: np.ndarray, pair_threshold: float) -> bool:
    """True only when every pair inside the group passes pair_threshold."""
    return all(float(sim[i, j]) >= float(pair_threshold) for i, j in combinations(sorted(group), 2))


# ---------------------------------------------------------------------------
# Debug formatting helpers
# ---------------------------------------------------------------------------

def _node_debug(nodes: list[NodeKey], i: int) -> NodeKey:
    return (int(nodes[i][0]), int(nodes[i][1]))


def _node_name(node: NodeKey) -> str:
    return f"E{int(node[0])}:T{int(node[1])}"


def _edge_debug(nodes: list[NodeKey], sim: np.ndarray, i: int, j: int, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "i": int(i),
        "j": int(j),
        "node_i": _node_debug(nodes, i),
        "node_j": _node_debug(nodes, j),
        "sim": float(sim[i, j]),
    }
    item.update(extra)
    return item


def _append_debug_edge(debug_list: list[dict[str, Any]], item: dict[str, Any], max_debug_edges: int) -> None:
    """Append a debug edge while keeping the debug payload bounded."""
    if len(debug_list) < max(0, int(max_debug_edges)):
        debug_list.append(item)


def _empty_rejection_counts() -> dict[str, int]:
    return {
        "pair_threshold_violation": 0,
        "same_epoch": 0,
        "epoch_conflict": 0,
        "merged_group_not_clique": 0,
    }


def _group_nodes_text(raw_group: Any) -> str:
    members = [(int(n[0]), int(n[1])) for n in (raw_group or [])]
    return ", ".join(_node_name(n) for n in members) if members else "EMPTY"


def format_global_clustering_debug_text(
    global_debug: dict[str, Any] | None,
    *,
    mapping: dict[NodeKey, int] | None = None,
    sim: np.ndarray | None = None,
    nodes: list[NodeKey] | np.ndarray | None = None,
) -> str:
    """Build a compact human-readable global clustering report.

    This is meant for saved TXT logs inside the global_clustering stage. It shows:
      - final global groups
      - invalid / unassigned tracks
      - accepted and rejected merge attempts
      - exact merge history: which temporary group absorbed which group
      - all pairwise similarity decisions
    """
    debug = global_debug or {}
    params = dict(debug.get("params", {}) or {})
    node_source = nodes if nodes is not None else debug.get("nodes", [])
    raw_nodes = node_source.tolist() if hasattr(node_source, "tolist") else node_source
    node_list: list[NodeKey] = [(int(n[0]), int(n[1])) for n in raw_nodes]
    sim_arr = None if sim is None else np.asarray(sim, dtype=np.float32)
    mapping_by_node = {(int(e), int(l)): int(g) for (e, l), g in (mapping or {}).items()}

    pair_threshold = float(params.get("pair_threshold", 0.95))
    allow_same_epoch = bool(params.get("allow_same_epoch_in_group", False))

    lines: list[str] = []
    lines.append("GLOBAL CLUSTERING DEBUG")
    lines.append("=" * 80)
    lines.append(f"method={debug.get('method', params.get('method', 'pairwise_clique'))}")
    lines.append(f"nodes={len(node_list)}")
    lines.append("")

    lines.append("PARAMS")
    lines.append("-" * 80)
    for key in sorted(params):
        lines.append(f"{key}={params[key]}")
    lines.append("")

    _append_groups_section(lines, debug)
    _append_invalid_groups_section(lines, debug)
    _append_unassigned_section(lines, debug)
    _append_rejection_counts_section(lines, debug)
    _append_merge_history_section(lines, debug)
    _append_accepted_edges_section(lines, debug)
    _append_rejected_edges_section(lines, debug)
    _append_all_pairwise_attempts_section(
        lines,
        node_list=node_list,
        sim_arr=sim_arr,
        mapping_by_node=mapping_by_node,
        pair_threshold=pair_threshold,
        allow_same_epoch=allow_same_epoch,
    )

    lines.append("")
    return "\n".join(lines) + "\n"


def _append_groups_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("FINAL GROUPS")
    lines.append("-" * 80)
    groups = list(debug.get("groups", []) or [])
    if not groups:
        lines.append("NO_VALID_GROUPS")
    for group in groups:
        gid = int(group.get("global_id", -1))
        members = [(int(n[0]), int(n[1])) for n in group.get("nodes", [])]
        members_txt = ", ".join(_node_name(n) for n in members)
        lines.append(
            f"G{gid} size={group.get('size', len(members))} "
            f"mean={float(group.get('mean_pairwise_sim', 0.0)):.4f} "
            f"min={float(group.get('min_pairwise_sim', 0.0)):.4f} | {members_txt}"
        )
    lines.append("")


def _append_invalid_groups_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("INVALID GROUPS")
    lines.append("-" * 80)
    invalid_groups = list(debug.get("invalid_groups", []) or [])
    if not invalid_groups:
        lines.append("NONE")
    for group in invalid_groups:
        members = [(int(n[0]), int(n[1])) for n in group.get("nodes", [])]
        members_txt = ", ".join(_node_name(n) for n in members)
        lines.append(
            f"reason={group.get('reason', 'unknown')} size={group.get('size', len(members))} "
            f"mean={float(group.get('mean_pairwise_sim', 0.0)):.4f} "
            f"min={float(group.get('min_pairwise_sim', 0.0)):.4f} | {members_txt}"
        )
    lines.append("")


def _append_unassigned_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("UNASSIGNED NODES")
    lines.append("-" * 80)
    unassigned = list(debug.get("unassigned_nodes", []) or [])
    if not unassigned:
        lines.append("NONE")
    for item in unassigned:
        node = item.get("node")
        if node is None:
            continue
        node_key = (int(node[0]), int(node[1]))
        lines.append(f"{_node_name(node_key)} reason={item.get('reason', 'unknown')}")
    lines.append("")


def _append_rejection_counts_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("REJECTION COUNTS")
    lines.append("-" * 80)
    counts = dict(debug.get("rejection_counts", {}) or {})
    if not counts:
        lines.append("NONE")
    for key in sorted(counts):
        lines.append(f"{key}={int(counts[key])}")
    lines.append("")


def _append_merge_history_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("MERGE HISTORY / WHO UPDATED TO WHOM")
    lines.append("-" * 80)
    accepted = list(debug.get("accepted_edges", []) or [])
    if not accepted:
        lines.append("NONE")
    for step, edge in enumerate(accepted, start=1):
        ni = (int(edge["node_i"][0]), int(edge["node_i"][1]))
        nj = (int(edge["node_j"][0]), int(edge["node_j"][1]))
        keep_id = edge.get("keep_group_id", "?")
        drop_id = edge.get("drop_group_id", "?")
        lines.append(
            f"STEP {step}: tempG{keep_id} <- tempG{drop_id} "
            f"because {_node_name(ni)} <-> {_node_name(nj)} sim={float(edge.get('sim', 0.0)):.4f}"
        )
        lines.append(f"  target_before: {_group_nodes_text(edge.get('group_keep_before'))}")
        lines.append(f"  source_added : {_group_nodes_text(edge.get('group_drop_before'))}")
        lines.append(f"  after        : {_group_nodes_text(edge.get('group_after'))}")
    lines.append("")


def _append_accepted_edges_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("ACCEPTED MERGE EDGES")
    lines.append("-" * 80)
    accepted = list(debug.get("accepted_edges", []) or [])
    if not accepted:
        lines.append("NONE")
    for edge in accepted:
        ni = (int(edge["node_i"][0]), int(edge["node_i"][1]))
        nj = (int(edge["node_j"][0]), int(edge["node_j"][1]))
        lines.append(
            f"PASS {_node_name(ni)} <-> {_node_name(nj)} "
            f"sim={float(edge.get('sim', 0.0)):.4f} merged_size={edge.get('merged_group_size', '?')} "
            f"tempG{edge.get('keep_group_id', '?')}<-tempG{edge.get('drop_group_id', '?')}"
        )
    lines.append("")


def _append_rejected_edges_section(lines: list[str], debug: dict[str, Any]) -> None:
    lines.append("REJECTED MERGE EDGES / EXAMPLES")
    lines.append("-" * 80)
    rejected = list(debug.get("rejected_edges", []) or [])
    if not rejected:
        lines.append("NONE")
    for edge in rejected:
        ni = (int(edge["node_i"][0]), int(edge["node_i"][1]))
        nj = (int(edge["node_j"][0]), int(edge["node_j"][1]))
        lines.append(
            f"REJECT {_node_name(ni)} <-> {_node_name(nj)} "
            f"sim={float(edge.get('sim', 0.0)):.4f} reason={edge.get('reason', 'unknown')}"
        )
    lines.append("")


def _append_all_pairwise_attempts_section(
    lines: list[str],
    *,
    node_list: list[NodeKey],
    sim_arr: np.ndarray | None,
    mapping_by_node: dict[NodeKey, int],
    pair_threshold: float,
    allow_same_epoch: bool,
) -> None:
    lines.append("ALL PAIRWISE ATTEMPTS")
    lines.append("-" * 80)
    if sim_arr is None or sim_arr.ndim != 2 or sim_arr.shape[0] != len(node_list):
        lines.append("NO_SIM_MATRIX_AVAILABLE")
        return

    pair_rows: list[tuple[float, int, int, str]] = []
    for i, j in combinations(range(len(node_list)), 2):
        score = float(sim_arr[i, j])
        ni, nj = node_list[i], node_list[j]
        gi = mapping_by_node.get(ni)
        gj = mapping_by_node.get(nj)
        same_epoch = int(ni[0]) == int(nj[0])

        if gi is not None and gi == gj:
            status = f"PASS_FINAL_G{gi}"
        elif score < pair_threshold:
            status = "REJECT_BELOW_THRESHOLD"
        elif same_epoch and not allow_same_epoch:
            status = "REJECT_SAME_EPOCH"
        else:
            status = "PASS_PAIR_NOT_FINAL_GROUP"

        pair_rows.append((score, i, j, status))

    pair_rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    for score, i, j, status in pair_rows:
        lines.append(f"{status} {_node_name(node_list[i])} <-> {_node_name(node_list[j])} sim={score:.4f}")


# ---------------------------------------------------------------------------
# Pairwise-clique global grouping
# ---------------------------------------------------------------------------

@dataclass
class _GroupingState:
    """Mutable state used while greedily merging compatible local-track groups."""
    groups_by_id: dict[int, set[int]]
    node_to_group: dict[int, int]
    accepted_edges: list[dict[str, Any]]

    @classmethod
    def singletons(cls, n_nodes: int) -> "_GroupingState":
        return cls(
            groups_by_id={i: {i} for i in range(n_nodes)},
            node_to_group={i: i for i in range(n_nodes)},
            accepted_edges=[],
        )


@dataclass
class GlobalTrackClusterer:
    """Conservative global ID builder using dense pairwise-compatible groups.

    Main idea:
        - each local track becomes one node
        - we connect only strong pairwise similarities
        - groups are valid only when every internal pair passes pair_threshold
        - uncertain tracks stay unassigned by default
    """

    method: str = "pairwise_clique"
    confirmed_only: bool = True
    min_track_history: int = 5
    pair_threshold: float = 0.95
    min_group_size: int = 3
    allow_same_epoch_in_group: bool = False
    assign_unknown: bool = False
    unknown_global_id: int = 999
    max_debug_edges: int = 200
    log_pair_threshold_rejections: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mapping(
        self,
        epoch_tracks: dict[int, dict[int, "Track"]],
        *,
        return_similarity: bool = False,
    ) -> dict[NodeKey, int] | tuple[dict[NodeKey, int], dict[str, list[NodeKey] | np.ndarray | dict]]:
        """Build ``(epoch_id, local_track_id) -> global_track_id`` mapping.

        High-level flow:
            1. Convert confirmed local tracks into one mean embedding per track.
            2. Build the track-to-track similarity matrix.
            3. Create candidate pair edges above pair_threshold.
            4. Greedily merge groups only when the merged group remains valid.
            5. Keep only groups with at least min_group_size nodes.
            6. Return mapping + debug/similarity payload.
        """
        nodes, sim = self._build_similarity_inputs(epoch_tracks)

        if not nodes:
            return self._return_empty_result(nodes, sim, return_similarity)

        debug = self._new_debug_containers()
        candidate_edges = self._build_candidate_edges(nodes, sim, debug)
        state = self._merge_candidate_edges(nodes, sim, candidate_edges, debug)
        valid_groups, invalid_groups = self._select_valid_groups(state.groups_by_id, nodes, sim)
        mapping, group_debug, unassigned_nodes = self._assign_global_ids(valid_groups, nodes, sim)

        if self.assign_unknown:
            self._assign_unknown_nodes(mapping, unassigned_nodes)

        global_debug = self._build_global_debug(
            nodes=nodes,
            sim=sim,
            debug=debug,
            accepted_edges=state.accepted_edges,
            groups=group_debug,
            invalid_groups=invalid_groups,
            unassigned_nodes=unassigned_nodes,
        )

        return self._return_result(mapping, nodes, sim, global_debug, return_similarity)

    # ------------------------------------------------------------------
    # Step 1-2: embeddings and similarity
    # ------------------------------------------------------------------

    def _build_similarity_inputs(self, epoch_tracks: dict[int, dict[int, "Track"]]) -> tuple[list[NodeKey], np.ndarray]:
        """Build nodes and their pairwise similarity matrix."""
        nodes, embs = build_mean_embeddings(
            epoch_tracks,
            confirmed_only=bool(self.confirmed_only),
            min_track_history=int(self.min_track_history),
        )
        return nodes, _cosine_similarity_matrix(embs)

    # ------------------------------------------------------------------
    # Step 3: pair candidates
    # ------------------------------------------------------------------

    def _new_debug_containers(self) -> dict[str, Any]:
        """Create mutable debug containers used during grouping."""
        return {
            "candidate_edges": [],
            "rejected_edges": [],
            "rejection_counts": _empty_rejection_counts(),
        }

    def _build_candidate_edges(
        self,
        nodes: list[NodeKey],
        sim: np.ndarray,
        debug: dict[str, Any],
    ) -> list[tuple[float, int, int]]:
        """Return candidate pair edges sorted from strongest to weakest.

        A candidate edge means:
            - pair similarity >= pair_threshold
            - and, unless explicitly allowed, the two nodes are not from same epoch
        """
        candidate_edges: list[tuple[float, int, int]] = []

        for i, j in combinations(range(len(nodes)), 2):
            score = float(sim[i, j])

            if not np.isfinite(score) or score < float(self.pair_threshold):
                self._record_rejection(debug, nodes, sim, i, j, "pair_threshold_violation", log=bool(self.log_pair_threshold_rejections))
                continue

            if self._same_epoch_pair_is_forbidden(nodes, i, j):
                self._record_rejection(debug, nodes, sim, i, j, "same_epoch")
                continue

            candidate_edges.append((score, i, j))
            _append_debug_edge(debug["candidate_edges"], _edge_debug(nodes, sim, i, j), int(self.max_debug_edges))

        candidate_edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
        return candidate_edges

    def _same_epoch_pair_is_forbidden(self, nodes: list[NodeKey], i: int, j: int) -> bool:
        return (not self.allow_same_epoch_in_group) and int(nodes[i][0]) == int(nodes[j][0])

    # ------------------------------------------------------------------
    # Step 4: merging
    # ------------------------------------------------------------------

    def _merge_candidate_edges(
        self,
        nodes: list[NodeKey],
        sim: np.ndarray,
        candidate_edges: list[tuple[float, int, int]],
        debug: dict[str, Any],
    ) -> _GroupingState:
        """Greedily merge candidate pairs while preserving group validity."""
        state = _GroupingState.singletons(len(nodes))

        for _score, i, j in candidate_edges:
            self._try_merge_edge(state, nodes, sim, i, j, debug)

        return state

    def _try_merge_edge(
        self,
        state: _GroupingState,
        nodes: list[NodeKey],
        sim: np.ndarray,
        i: int,
        j: int,
        debug: dict[str, Any],
    ) -> None:
        """Try to merge the two current groups containing node i and node j."""
        group_i = state.node_to_group[i]
        group_j = state.node_to_group[j]
        if group_i == group_j:
            return

        keep_id = min(group_i, group_j)
        drop_id = max(group_i, group_j)
        merged = set(state.groups_by_id[keep_id]) | set(state.groups_by_id[drop_id])

        if self._merged_group_has_forbidden_epoch_conflict(merged, nodes):
            self._record_rejection(debug, nodes, sim, i, j, "epoch_conflict")
            return

        if not _is_strict_clique(merged, sim, float(self.pair_threshold)):
            self._record_rejection(debug, nodes, sim, i, j, "merged_group_not_clique")
            return

        self._accept_merge(state, nodes, sim, i, j, keep_id, drop_id, merged)

    def _merged_group_has_forbidden_epoch_conflict(self, merged: set[int], nodes: list[NodeKey]) -> bool:
        return (not self.allow_same_epoch_in_group) and _has_epoch_conflict(merged, nodes)

    def _accept_merge(
        self,
        state: _GroupingState,
        nodes: list[NodeKey],
        sim: np.ndarray,
        i: int,
        j: int,
        keep_id: int,
        drop_id: int,
        merged: set[int],
    ) -> None:
        """Commit a successful merge and save a readable merge-history record."""
        keep_before = sorted(int(x) for x in state.groups_by_id[keep_id])
        drop_before = sorted(int(x) for x in state.groups_by_id[drop_id])
        after = sorted(int(x) for x in merged)

        state.groups_by_id[keep_id] = merged
        del state.groups_by_id[drop_id]

        for member in merged:
            state.node_to_group[member] = keep_id

        _append_debug_edge(
            state.accepted_edges,
            _edge_debug(
                nodes,
                sim,
                i,
                j,
                merged_group_size=len(merged),
                keep_group_id=int(keep_id),
                drop_group_id=int(drop_id),
                group_keep_before=[_node_debug(nodes, x) for x in keep_before],
                group_drop_before=[_node_debug(nodes, x) for x in drop_before],
                group_after=[_node_debug(nodes, x) for x in after],
            ),
            int(self.max_debug_edges),
        )

    # ------------------------------------------------------------------
    # Step 5: keep only valid global groups
    # ------------------------------------------------------------------

    def _select_valid_groups(
        self,
        groups_by_id: dict[int, set[int]],
        nodes: list[NodeKey],
        sim: np.ndarray,
    ) -> tuple[list[set[int]], list[dict[str, Any]]]:
        """Split temporary groups into final valid groups and invalid leftovers."""
        all_groups = [set(group) for _gid, group in sorted(groups_by_id.items(), key=lambda item: min(item[1]))]
        valid_groups: list[set[int]] = []
        invalid_groups: list[dict[str, Any]] = []

        for group in all_groups:
            if len(group) >= int(self.min_group_size):
                valid_groups.append(set(group))
            else:
                invalid_groups.append(self._invalid_group_debug(group, nodes, sim, reason="too_small"))

        valid_groups.sort(
            key=lambda g: (
                -len(g),
                -_group_mean_similarity(g, sim),
                min(_node_debug(nodes, i) for i in g),
            )
        )
        return valid_groups, invalid_groups

    def _invalid_group_debug(
        self,
        group: set[int] | list[int],
        nodes: list[NodeKey],
        sim: np.ndarray,
        *,
        reason: str,
    ) -> dict[str, Any]:
        idxs = sorted(int(i) for i in group)
        return {
            "node_indices": idxs,
            "nodes": [_node_debug(nodes, i) for i in idxs],
            "reason": reason,
            "size": len(idxs),
            "mean_pairwise_sim": _group_mean_similarity(idxs, sim),
            "min_pairwise_sim": _group_min_similarity(idxs, sim),
        }

    # ------------------------------------------------------------------
    # Step 6: final mapping/debug
    # ------------------------------------------------------------------

    def _assign_global_ids(
        self,
        valid_groups: list[set[int]],
        nodes: list[NodeKey],
        sim: np.ndarray,
    ) -> tuple[dict[NodeKey, int], list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert valid groups into final global IDs and collect leftovers."""
        mapping: dict[NodeKey, int] = {}
        group_debug: list[dict[str, Any]] = []
        assigned_indices: set[int] = set()

        for global_id, group in enumerate(valid_groups, start=1):
            idxs = sorted(int(i) for i in group)
            for i in idxs:
                mapping[_node_debug(nodes, i)] = int(global_id)
                assigned_indices.add(i)
            group_debug.append(self._valid_group_debug(global_id, idxs, nodes, sim))

        unassigned_nodes = [
            {"i": int(i), "node": _node_debug(nodes, i), "reason": "not_in_valid_group"}
            for i in range(len(nodes))
            if i not in assigned_indices
        ]
        return mapping, group_debug, unassigned_nodes

    def _valid_group_debug(
        self,
        global_id: int,
        idxs: list[int],
        nodes: list[NodeKey],
        sim: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "global_id": int(global_id),
            "node_indices": idxs,
            "nodes": [_node_debug(nodes, i) for i in idxs],
            "size": len(idxs),
            "mean_pairwise_sim": _group_mean_similarity(idxs, sim),
            "min_pairwise_sim": _group_min_similarity(idxs, sim),
        }

    def _assign_unknown_nodes(self, mapping: dict[NodeKey, int], unassigned_nodes: list[dict[str, Any]]) -> None:
        """Optionally write all leftovers to a configured unknown global ID."""
        for item in unassigned_nodes:
            node = item.get("node")
            if node is not None:
                mapping[(int(node[0]), int(node[1]))] = int(self.unknown_global_id)

    def _build_global_debug(
        self,
        *,
        nodes: list[NodeKey],
        sim: np.ndarray,
        debug: dict[str, Any],
        accepted_edges: list[dict[str, Any]],
        groups: list[dict[str, Any]],
        invalid_groups: list[dict[str, Any]],
        unassigned_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "method": "pairwise_clique",
            "params": self._params_debug(),
            "nodes": [tuple(n) for n in nodes],
            "sim_shape": tuple(int(x) for x in sim.shape),
            "rejection_counts": debug["rejection_counts"],
            "candidate_edges": debug["candidate_edges"],
            "accepted_edges": accepted_edges,
            "rejected_edges": debug["rejected_edges"],
            "groups": groups,
            "invalid_groups": invalid_groups,
            "unassigned_nodes": unassigned_nodes,
        }

    def _return_empty_result(
        self,
        nodes: list[NodeKey],
        sim: np.ndarray,
        return_similarity: bool,
    ) -> dict[NodeKey, int] | tuple[dict[NodeKey, int], dict[str, list[NodeKey] | np.ndarray | dict]]:
        mapping: dict[NodeKey, int] = {}
        global_debug = self._empty_debug(nodes, sim)
        return self._return_result(mapping, nodes, sim, global_debug, return_similarity)

    def _return_result(
        self,
        mapping: dict[NodeKey, int],
        nodes: list[NodeKey],
        sim: np.ndarray,
        global_debug: dict[str, Any],
        return_similarity: bool,
    ) -> dict[NodeKey, int] | tuple[dict[NodeKey, int], dict[str, list[NodeKey] | np.ndarray | dict]]:
        if return_similarity:
            return mapping, {"nodes": nodes, "sim": sim, "global_debug": global_debug}
        return mapping

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _params_debug(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "confirmed_only": bool(self.confirmed_only),
            "min_track_history": int(self.min_track_history),
            "pair_threshold": float(self.pair_threshold),
            "min_group_size": int(self.min_group_size),
            "allow_same_epoch_in_group": bool(self.allow_same_epoch_in_group),
            "assign_unknown": bool(self.assign_unknown),
            "unknown_global_id": int(self.unknown_global_id),
            "max_debug_edges": int(self.max_debug_edges),
            "log_pair_threshold_rejections": bool(self.log_pair_threshold_rejections),
        }

    def _empty_debug(self, nodes: list[NodeKey], sim: np.ndarray) -> dict[str, Any]:
        return {
            "method": "pairwise_clique",
            "params": self._params_debug(),
            "nodes": [tuple(n) for n in nodes],
            "sim_shape": tuple(int(x) for x in sim.shape),
            "rejection_counts": _empty_rejection_counts(),
            "candidate_edges": [],
            "accepted_edges": [],
            "rejected_edges": [],
            "groups": [],
            "invalid_groups": [],
            "unassigned_nodes": [
                {"i": int(i), "node": _node_debug(nodes, i), "reason": "not_in_valid_group"}
                for i in range(len(nodes))
            ],
        }

    def _record_rejection(
        self,
        debug: dict[str, Any],
        nodes: list[NodeKey],
        sim: np.ndarray,
        i: int,
        j: int,
        reason: str,
        *,
        log: bool = True,
    ) -> None:
        """Count all rejections, but store only a capped number of examples."""
        debug["rejection_counts"][reason] = int(debug["rejection_counts"].get(reason, 0)) + 1
        if log:
            _append_debug_edge(
                debug["rejected_edges"],
                _edge_debug(nodes, sim, i, j, reason=reason),
                int(self.max_debug_edges),
            )