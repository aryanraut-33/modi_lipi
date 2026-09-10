import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from .glyph_dataclass import SegmentedGlyph
    from .character_segmenter import CharacterSegmenter, CharSegConfig
    from .ai_glyph_validator import AIGlyphValidator
except ImportError:
    from glyph_dataclass import SegmentedGlyph
    from character_segmenter import CharacterSegmenter, CharSegConfig
    from ai_glyph_validator import AIGlyphValidator

class AgenticHybridSegmenter:
    """
    Hybrid Dual-Engine Character Segmentation Orchestrator for Modi ShatVidhi.
    Supports:
      1. Pure fast algorithmic segmentation (Phase 3 baseline across entire document).
      2. Interactive on-demand single-glyph or whole-line AI verification (via local Ollama or Gemini).
    """

    def __init__(
        self,
        base_config: Optional[CharSegConfig] = None,
        enable_ai_validator: bool = True,
        max_ai_iterations: int = 2
    ):
        self.base_segmenter = CharacterSegmenter(config=base_config)
        self.enable_ai_validator = enable_ai_validator
        self.max_ai_iterations = max_ai_iterations
        self.ai_validator = AIGlyphValidator() if enable_ai_validator else None

    def segment_line_algorithmic(
        self,
        line_img: np.ndarray,
        line_id: int = 0
    ) -> Tuple[List[SegmentedGlyph], Dict[str, Any]]:
        """
        Fast 100% local algorithmic segmentation across the line (0 token cost, <5ms).
        """
        return self.base_segmenter.segment_line(line_img, line_id=line_id)

    def reverify_single_glyph(
        self,
        line_img: np.ndarray,
        current_glyphs: List[SegmentedGlyph],
        glyph_idx: int,
        line_id: int = 0
    ) -> Tuple[List[SegmentedGlyph], Dict[str, Any]]:
        """
        Interactive On-Demand AI Fix:
        Sends ONLY the user-selected glyph (and line overview) to the local Ollama/Gemini model,
        applies the corrective action, and returns the updated glyph sequence.
        """
        if not current_glyphs or glyph_idx < 0 or glyph_idx >= len(current_glyphs) or not self.ai_validator:
            return current_glyphs, {"action": "NO_CHANGE", "reason": "Invalid index or validator unavailable"}

        h, w = line_img.shape[:2]
        gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) if len(line_img.shape) == 3 else line_img.copy()
        _, bin_inv = cv2.threshold(gray_line, 200, 255, cv2.THRESH_BINARY_INV)

        current_spans = [(g.bbox[0], g.bbox[2]) for g in current_glyphs]
        ls = current_glyphs[0].metadata.get("shirorekha_y", int(h * 0.35))
        lb = current_glyphs[0].metadata.get("baseline_y", int(h * 0.75))

        # Build annotated line overview
        annotated_line = cv2.cvtColor(gray_line, cv2.COLOR_GRAY2RGB)
        cv2.line(annotated_line, (0, ls), (w, ls), (0, 180, 255), 1)
        cv2.line(annotated_line, (0, lb), (w, lb), (255, 120, 0), 1)

        for idx, (x1, x2) in enumerate(current_spans):
            color = (0, 0, 255) if idx == glyph_idx else (0, 200, 0)
            thick = 2 if idx == glyph_idx else 1
            cv2.rectangle(annotated_line, (x1, 0), (x2, h), color, thick)
            cv2.putText(annotated_line, str(idx + 1), (x1 + 2, min(h - 4, ls + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        target_crop = gray_line[:, current_spans[glyph_idx][0]:current_spans[glyph_idx][1]]

        # Call AI Validator for this single glyph
        res = self.ai_validator.validate_single_glyph(
            annotated_line, target_crop, glyph_idx=glyph_idx, total_glyphs=len(current_glyphs)
        )

        action = str(res.get("action", "ACCEPT")).upper()
        reason = res.get("reason", "")
        new_spans = list(current_spans)

        # Apply correction
        if "MERGE_LEFT" in action and glyph_idx > 0:
            prev_x1, _ = new_spans[glyph_idx - 1]
            curr_x1, curr_x2 = new_spans[glyph_idx]
            new_spans[glyph_idx - 1] = (prev_x1, curr_x2)
            del new_spans[glyph_idx]
        elif "MERGE_RIGHT" in action and glyph_idx < len(new_spans) - 1:
            curr_x1, curr_x2 = new_spans[glyph_idx]
            next_x1, next_x2 = new_spans[glyph_idx + 1]
            new_spans[glyph_idx] = (curr_x1, next_x2)
            del new_spans[glyph_idx + 1]
        elif "SPLIT" in action:
            curr_x1, curr_x2 = new_spans[glyph_idx]
            split_ratio = float(res.get("split_ratio", 0.50))
            split_ratio = max(0.2, min(0.8, split_ratio))
            sub_w = curr_x2 - curr_x1
            mid_split = curr_x1 + int(sub_w * split_ratio)

            body_sub = bin_inv[ls + 2:lb, max(0, mid_split - 8):min(w, mid_split + 8)]
            if body_sub.size > 0:
                sub_vpp = np.sum(body_sub == 255, axis=0)
                best_offset = int(np.argmin(sub_vpp)) - 8
                mid_split = max(curr_x1 + 8, min(curr_x2 - 8, mid_split + best_offset))

            new_spans[glyph_idx] = (curr_x1, mid_split)
            new_spans.insert(glyph_idx + 1, (mid_split, curr_x2))

        # Re-build finalized SegmentedGlyph objects
        updated_glyphs = self._build_glyph_objects(gray_line, bin_inv, new_spans, line_id, ls, lb)
        return updated_glyphs, {"action": action, "reason": reason, "target_idx": glyph_idx}

    def _build_glyph_objects(
        self,
        gray_line: np.ndarray,
        bin_inv: np.ndarray,
        spans: List[Tuple[int, int]],
        line_id: int,
        ls: int,
        lb: int
    ) -> List[SegmentedGlyph]:
        h, w = gray_line.shape[:2]
        glyphs = []
        for g_idx, (x1, x2) in enumerate(spans):
            if x2 <= x1:
                continue

            char_bin_slice = bin_inv[:, x1:x2]
            row_sums = np.sum(char_bin_slice == 255, axis=1)
            non_zero_rows = np.where(row_sums > 0)[0]
            if len(non_zero_rows) > 0:
                y_min = max(0, non_zero_rows[0] - 2)
                y_max = min(h, non_zero_rows[-1] + 3)
            else:
                y_min, y_max = 0, h

            full_crop = gray_line[y_min:y_max, x1:x2]
            if full_crop.size == 0:
                full_crop = gray_line[:, x1:x2]

            base_crop = gray_line[ls:lb, x1:x2] if lb > ls else full_crop
            upper_crop = gray_line[0:ls, x1:x2] if ls > 0 else None
            lower_crop = gray_line[lb:h, x1:x2] if lb < h else None

            upper_ink = np.sum(bin_inv[0:ls, x1:x2] == 255) if ls > 0 else 0
            lower_ink = np.sum(bin_inv[lb:h, x1:x2] == 255) if lb < h else 0

            glyph_w = x2 - x1
            glyph_h = y_max - y_min
            aspect_ratio = float(glyph_w / max(1, glyph_h))
            density = float(np.sum(char_bin_slice == 255) / max(1, glyph_h * glyph_w))

            glyph_obj = SegmentedGlyph(
                glyph_id=g_idx,
                line_id=line_id,
                bbox=(x1, y_min, x2, y_max),
                full_glyph_img=full_crop,
                base_glyph_img=base_crop,
                upper_zone_img=upper_crop,
                lower_zone_img=lower_crop,
                has_upper_modifier=bool(upper_ink >= 12),
                has_lower_modifier=bool(lower_ink >= 12),
                aspect_ratio=aspect_ratio,
                density=density,
                metadata={
                    "shirorekha_y": ls,
                    "baseline_y": lb,
                    "target_size": (64, 64)
                }
            )
            glyphs.append(glyph_obj)

        return glyphs
