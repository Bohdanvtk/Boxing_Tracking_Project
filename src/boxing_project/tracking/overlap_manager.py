from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .track import Detection, Track


class OverlapManager:
    """Overlap-risk and appearance-freeze rules used by MultiObjectTracker."""

    def __init__(self, cfg: Any):
        self.cfg = cfg

    def build_det_idx_to_track(
        self,
        tracks: List[Track],
        matches_idx: List[Tuple[int, int]],
    ) -> Dict[int, Track]:
        """Return det_idx -> Track for current matches."""
        return {
            int(j_det): tracks[int(i_track)]
            for i_track, j_det in matches_idx
        }

    def track_stable(self, track: Optional[Track]) -> bool:
        return bool(track is not None and (track.sub_confirmed or track.confirmed))

    def pair_overlap_threshold(
        self,
        current_track: Track,
        other_track: Optional[Track],
        center_dist_norm: float,
    ) -> Tuple[float, str, str]:
        """Choose adaptive IoU threshold for one overlap pair."""
        current_stable = self.track_stable(current_track)
        other_stable = self.track_stable(other_track)

        if not current_stable:
            return self.cfg.adaptive_overlap_iou_default, "default", "current_track_pending_default_only"
        if not other_stable:
            return self.cfg.adaptive_overlap_iou_default, "default", "other_track_pending_or_unmatched_default_only"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_near:
            return self.cfg.adaptive_overlap_iou_near, "near", "stable_pair_adaptive"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_mid:
            return self.cfg.adaptive_overlap_iou_mid, "mid", "stable_pair_adaptive"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_far:
            return self.cfg.adaptive_overlap_iou_far, "far", "stable_pair_adaptive"
        return self.cfg.adaptive_overlap_iou_default, "default", "stable_pair_default_far"

    def evaluate_detection_overlap(
        self,
        trk: Track,
        det: Detection,
        det_idx_to_track: Optional[Dict[int, Track]] = None,
    ) -> Dict[str, Any]:
        """Evaluate overlap risk and enrich det.meta relations."""
        relations = det.meta.get("overlap_relations", []) or []
        risky_indices: List[int] = []
        risky_ious: List[float] = []
        best_any_rel: Optional[Dict[str, Any]] = None
        best_risky_rel: Optional[Dict[str, Any]] = None
        overlap_has_stable_track = False

        for rel in relations:
            other_idx = rel.get("det_idx")
            other_track = (
                det_idx_to_track.get(int(other_idx))
                if det_idx_to_track is not None and other_idx is not None
                else None
            )
            other_stable = self.track_stable(other_track)
            overlap_has_stable_track = overlap_has_stable_track or other_stable
            cdn = float(rel.get("center_dist_norm", det.meta.get("min_center_dist_norm", float("inf"))))
            threshold, zone, reason = self.pair_overlap_threshold(trk, other_track, cdn)
            iou = float(rel.get("iou", 0.0))
            risky = bool(iou > float(threshold))

            rel["adaptive_overlap_threshold"] = float(threshold)
            rel["adaptive_overlap_zone"] = zone
            rel["adaptive_overlap_reason"] = reason
            rel["adaptive_overlap_risk"] = risky
            rel["other_track_stable"] = bool(other_stable)

            if best_any_rel is None or iou > float(best_any_rel.get("iou", 0.0)):
                best_any_rel = rel
            if risky:
                if best_risky_rel is None or iou > float(best_risky_rel.get("iou", 0.0)):
                    best_risky_rel = rel
                if other_idx is not None:
                    risky_indices.append(int(other_idx))
                risky_ious.append(iou)

        fallback = {
            "adaptive_overlap_threshold": float(self.cfg.adaptive_overlap_iou_default),
            "adaptive_overlap_zone": "default",
            "adaptive_overlap_reason": "no_overlap_relations_default_only",
        }
        active = best_risky_rel or best_any_rel or fallback
        return {
            "has_overlap": bool(risky_ious),
            "current_track_stable": self.track_stable(trk),
            "overlap_has_stable_track": bool(overlap_has_stable_track),
            "active_overlap_threshold": float(active.get("adaptive_overlap_threshold", self.cfg.adaptive_overlap_iou_default)),
            "adaptive_overlap_zone": active.get("adaptive_overlap_zone", "default"),
            "adaptive_overlap_reason": active.get("adaptive_overlap_reason", "no_overlap_relations_default_only"),
            "adaptive_overlap_enabled": str(active.get("adaptive_overlap_reason", "")).startswith("stable_pair"),
            "risky_overlap_count": len(risky_indices),
            "risky_overlap_det_indices": risky_indices,
            "max_risky_overlap_iou": max(risky_ious) if risky_ious else 0.0,
        }

    def prepare_overlap_update_meta(
        self,
        trk: Track,
        det: Detection,
        det_idx_to_track: Optional[Dict[int, Track]] = None,
    ) -> bool:
        """Write overlap decision into det.meta before Track.update()."""
        eval_meta = self.evaluate_detection_overlap(trk, det, det_idx_to_track)
        raw_max_overlap_iou = float(det.meta.get("max_overlap_iou", 0.0))
        has_overlap = bool(eval_meta["has_overlap"])

        det.meta["track_hits_before_update"] = int(trk.hits)
        det.meta["track_sub_confirmed_before_update"] = bool(trk.sub_confirmed)
        det.meta["track_confirmed_before_update"] = bool(trk.confirmed)
        det.meta["track_has_risky_overlap"] = has_overlap
        det.meta["is_overlapping"] = has_overlap
        det.meta["adaptive_overlap_enabled"] = bool(eval_meta["adaptive_overlap_enabled"])
        det.meta["adaptive_overlap_reason"] = eval_meta["adaptive_overlap_reason"]
        det.meta["adaptive_overlap_zone"] = eval_meta["adaptive_overlap_zone"]
        det.meta["active_overlap_threshold"] = float(eval_meta["active_overlap_threshold"])
        det.meta["risky_overlap_count"] = int(eval_meta["risky_overlap_count"])
        det.meta["risky_overlap_det_indices"] = list(eval_meta["risky_overlap_det_indices"])
        det.meta["max_risky_overlap_iou"] = float(eval_meta["max_risky_overlap_iou"])
        det.meta["raw_max_overlap_iou"] = raw_max_overlap_iou
        det.meta["max_overlap_iou"] = raw_max_overlap_iou
        det.meta["max_overlap_det_idx"] = det.meta.get("max_overlap_det_idx")
        det.meta["min_center_dist_norm"] = float(det.meta.get("min_center_dist_norm", float("inf")))
        det.meta["center_dist_norm_det_idx"] = det.meta.get("center_dist_norm_det_idx")
        det.meta["current_track_stable_for_overlap"] = bool(eval_meta["current_track_stable"])
        det.meta["overlap_has_stable_track"] = bool(eval_meta["overlap_has_stable_track"])
        return has_overlap

    def compare_matches(
        self,
        prev_match: Dict[int, int],
        current_match: Dict[int, int],
    ) -> set[int]:
        """Return tracks that were matched in the previous frame but not now."""
        return set(prev_match.keys()) - set(current_match.keys())

    def dets_to_track(
        self,
        matches: Dict[int, int],
        detections: List[Detection],
    ) -> Dict[int, set[int]]:
        """Convert detection-overlap groups into track-id groups."""
        det_to_track = {
            int(det_idx): int(track_id)
            for track_id, det_idx in matches.items()
        }

        track_groups: Dict[int, set[int]] = {}

        for track_id, det_idx in matches.items():
            track_id = int(track_id)
            det_idx = int(det_idx)

            if not (0 <= det_idx < len(detections)):
                track_groups[track_id] = set()
                continue

            det = detections[det_idx]
            overlap_det_indices = Track.overlap_group(det)
            group_track_ids: set[int] = set()

            for other_det_idx in overlap_det_indices:
                other_track_id = det_to_track.get(int(other_det_idx))
                if other_track_id is None:
                    continue
                if int(other_track_id) == track_id:
                    continue
                group_track_ids.add(int(other_track_id))

            track_groups[track_id] = group_track_ids

        return track_groups

    def whether_in_group(
        self,
        idx: int,
        groups: Dict[int, set[int]],
    ) -> set[int]:
        """Return all track ids that are in the same overlap group as idx."""
        idx = int(idx)
        result: set[int] = set()

        for owner_id, members in groups.items():
            full_group = {int(owner_id)} | {int(x) for x in members}
            if idx in full_group:
                result |= full_group

        result.discard(idx)
        return result

    def set_cooldown(
        self,
        track: Track,
        source_track_id: int,
    ) -> None:
        """Set appearance freeze on one track because source_track_id disappeared."""
        track.set_cooldown(
            source_track_id=int(source_track_id),
            frames=self.cfg.overlap_app_freeze_after,
        )

    def freeze_track_near_unmatched(
        self,
        absent_ids: set[int],
        tracks: List[Track],
    ) -> set[int]:
        """
        Freeze tracks that were in an overlap group with disappeared sources.

        If source track M disappeared and another track had M in
        overlap_group_ids, freeze that other track with source M. Also freeze M
        itself if it still exists in the active track list.
        """
        tracks_by_id = {
            int(track.track_id): track
            for track in tracks
        }

        freshly_frozen_sources: set[int] = set()

        for absent_id in {int(x) for x in absent_ids}:
            affected_ids = {absent_id}

            for track in tracks:
                if absent_id in getattr(track, "overlap_group_ids", set()):
                    affected_ids.add(int(track.track_id))

            for affected_id in affected_ids:
                track = tracks_by_id.get(int(affected_id))
                if track is None:
                    continue
                self.set_cooldown(
                    track=track,
                    source_track_id=absent_id,
                )

            if affected_ids:
                freshly_frozen_sources.add(absent_id)

        return freshly_frozen_sources

    def which_defroze(
        self,
        matched_track_ids: set[int],
        tracks: List[Track],
    ) -> None:
        """
        Clear freeze source M globally only when source track M matched again.

        Track N matching does not clear freeze caused by M; only M returning does.
        """
        for source_track_id in {int(x) for x in matched_track_ids}:
            for track in tracks:
                track.clear_freeze_source(source_track_id)

    def decrease_freeze(
        self,
        tracks: List[Track],
        exclude_sources: Optional[set[int]] = None,
    ) -> None:
        """
        Decrease all active freezes by one frame.

        Sources created on the current frame can be excluded, so a fresh cooldown
        does not immediately lose one frame.
        """
        exclude_sources = {int(x) for x in (exclude_sources or set())}

        for track in tracks:
            track.decrease_freeze(exclude_sources=exclude_sources)
