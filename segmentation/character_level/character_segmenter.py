import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

try:
    from .glyph_dataclass import SegmentedGlyph
    from .zonal_analyzer import ZonalAnalyzer
    from .ligature_engine import LigatureEngine
    from .cluster_splitter import ClusterSplitter
    from .postprocessor import PostProcessor
except ImportError:
    from glyph_dataclass import SegmentedGlyph
    from zonal_analyzer import ZonalAnalyzer
    from ligature_engine import LigatureEngine
    from cluster_splitter import ClusterSplitter
    from postprocessor import PostProcessor

@dataclass
class CharSegConfig:
    """
    Tuneable hyperparameters for Phase 3 Glyph-Level Character Segmentation.
    """
    # 1. Zonal Parameters
    upper_search_ratio: float = 0.50   # Search window for Shirorekha peak (top 50%)
    lower_search_ratio: float = 0.65   # Search window for Baseline valley

    # 2. Ligature Candidate Parameters
    min_char_width: int = 10           # Minimum candidate character width
    vpp_valley_factor: float = 0.65    # Ratio against mean body density to detect valleys

    # 3. Touching Cluster Parameters
    cluster_width_multiplier: float = 1.60 # Multiplier above median width to trigger cluster splitting
    min_split_width: int = 12          # Minimum width of child cluster segments

    # 4. Modi Syllable & Post-processing Parameters
    intensity_merge_ratio: float = 0.30 # Multiplier for mean(IntSeg) merge threshold
    min_ink_pixels: int = 20           # Minimum ink pixels to qualify as a valid character
    max_kana_width: int = 18           # Maximum width of trailing vertical stem/kana
    
    # 5. Output Normalization
    target_glyph_size: Tuple[int, int] = (64, 64) # Fixed tensor size for Phase 4 recognition

class CharacterSegmenter:
    """
    High-level orchestrator for Phase 3 (Glyph-Level Character Segmentation) of Modi ShatVidhi.
    Segments preprocessed text lines into complete, grammatically fused Modi glyph units.
    """

    def __init__(self, config: Optional[CharSegConfig] = None):
        self.cfg = config or CharSegConfig()
        self.zonal_analyzer = ZonalAnalyzer(
            upper_search_ratio=self.cfg.upper_search_ratio,
            lower_search_ratio=self.cfg.lower_search_ratio
        )
        self.ligature_engine = LigatureEngine(
            min_char_width=self.cfg.min_char_width,
            vpp_valley_factor=self.cfg.vpp_valley_factor
        )
        self.cluster_splitter = ClusterSplitter(
            cluster_width_multiplier=self.cfg.cluster_width_multiplier,
            min_split_width=self.cfg.min_split_width
        )
        self.postprocessor = PostProcessor(
            intensity_merge_ratio=self.cfg.intensity_merge_ratio,
            min_ink_pixels=self.cfg.min_ink_pixels,
            max_kana_width=self.cfg.max_kana_width
        )

    def segment_line(self, line_img: np.ndarray, line_id: int = 0) -> Tuple[List[SegmentedGlyph], Dict[str, Any]]:
        """
        Segments a single text line into complete individual character glyphs.

        Args:
            line_img: 2D or 3D numpy array of the text line.
            line_id: Index of the current line.

        Returns:
            Tuple of:
              - List of SegmentedGlyph objects
              - Debug/intermediate metadata dictionary
        """
        if line_img is None or line_img.size == 0:
            return [], {}

        h, w = line_img.shape[:2]
        if h < 10 or w < 10:
            return [], {}

        # 1. Grayscale & Inverted Binary Representation
        if len(line_img.shape) == 3:
            gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_line = line_img.copy()

        _, bin_inv = cv2.threshold(gray_line, 200, 255, cv2.THRESH_BINARY_INV)

        # 2. Step 1 & 2: Zonal Analysis (Shirorekha & Baseline Detection)
        zonal_info = self.zonal_analyzer.analyze_line(gray_line)
        shirorekha_y = zonal_info["shirorekha_y"]
        baseline_y = zonal_info["baseline_y"]

        # 3. Step 3: Ligature & Boundary Sweep (Middle Zone + Upper Matra Protection)
        psc_candidates = self.ligature_engine.find_candidate_columns(
            gray_line, bin_inv, shirorekha_y, baseline_y
        )

        # 4. Step 4: Touching & Overlapping Cluster Splitting
        split_candidates = self.cluster_splitter.evaluate_and_split_clusters(
            psc_candidates, gray_line[shirorekha_y:baseline_y, :], bin_inv
        )

        # 5. Step 5: Modi Syllable Binding & Post-Processing (FSC)
        fsc_pairs = self.postprocessor.finalize_columns(
            split_candidates, bin_inv, shirorekha_y, baseline_y
        )

        # 6. Extract Complete Vertical Envelope for Every Character
        glyphs = []
        for g_idx, (x1, x2) in enumerate(fsc_pairs):
            if x2 <= x1:
                continue

            char_bin_slice = bin_inv[:, x1:x2]
            
            # Find vertical ink extent [y_min, y_max] for this character column
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

            base_crop = gray_line[shirorekha_y:baseline_y, x1:x2] if baseline_y > shirorekha_y else full_crop
            upper_crop = gray_line[0:shirorekha_y, x1:x2] if shirorekha_y > 0 else None
            lower_crop = gray_line[baseline_y:h, x1:x2] if baseline_y < h else None

            upper_ink = np.sum(bin_inv[0:shirorekha_y, x1:x2] == 255) if shirorekha_y > 0 else 0
            lower_ink = np.sum(bin_inv[baseline_y:h, x1:x2] == 255) if baseline_y < h else 0
            has_upper = bool(upper_ink >= 12)
            has_lower = bool(lower_ink >= 12)

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
                has_upper_modifier=has_upper,
                has_lower_modifier=has_lower,
                aspect_ratio=aspect_ratio,
                density=density,
                metadata={
                    "shirorekha_y": shirorekha_y,
                    "baseline_y": baseline_y,
                    "target_size": self.cfg.target_glyph_size
                }
            )
            glyphs.append(glyph_obj)

        debug_info = {
            "zonal_info": zonal_info,
            "psc_candidates": psc_candidates,
            "split_candidates": split_candidates,
            "fsc_pairs": fsc_pairs,
            "num_glyphs": len(glyphs)
        }

        return glyphs, debug_info

    def merge_glyph_spans(
        self,
        line_img: np.ndarray,
        current_glyphs: List[SegmentedGlyph],
        indices_to_merge: List[int],
        line_id: int = 0
    ) -> List[SegmentedGlyph]:
        """
        Merges selected adjacent glyphs into single consolidated SegmentedGlyph objects.
        """
        if not current_glyphs or not indices_to_merge:
            return current_glyphs

        indices = sorted(set(indices_to_merge))
        h, w = line_img.shape[:2]
        gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) if len(line_img.shape) == 3 else line_img.copy()
        _, bin_inv = cv2.threshold(gray_line, 200, 255, cv2.THRESH_BINARY_INV)

        ls = current_glyphs[0].metadata.get("shirorekha_y", int(h * 0.35))
        lb = current_glyphs[0].metadata.get("baseline_y", int(h * 0.75))

        # Identify contiguous groups in indices
        groups = []
        curr_group = [indices[0]]
        for idx in indices[1:]:
            if idx == curr_group[-1] + 1:
                curr_group.append(idx)
            else:
                groups.append(curr_group)
                curr_group = [idx]
        groups.append(curr_group)

        # Build new spans
        old_spans = [(g.bbox[0], g.bbox[2]) for g in current_glyphs]
        new_spans = []
        skip_set = set(indices)

        for i in range(len(current_glyphs)):
            if i in skip_set:
                # Check if i is the start of any group
                for grp in groups:
                    if grp[0] == i:
                        merged_x1 = old_spans[grp[0]][0]
                        merged_x2 = old_spans[grp[-1]][1]
                        new_spans.append((merged_x1, merged_x2))
                        break
            else:
                new_spans.append(old_spans[i])

        return self._build_glyph_objects(gray_line, bin_inv, new_spans, line_id, ls, lb)

    def split_glyph_span(
        self,
        line_img: np.ndarray,
        current_glyphs: List[SegmentedGlyph],
        glyph_idx: int,
        split_ratio: float = 0.50,
        line_id: int = 0
    ) -> List[SegmentedGlyph]:
        """
        Splits an under-segmented cluster into two individual character glyphs.
        """
        if not current_glyphs or glyph_idx < 0 or glyph_idx >= len(current_glyphs):
            return current_glyphs

        h, w = line_img.shape[:2]
        gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) if len(line_img.shape) == 3 else line_img.copy()
        _, bin_inv = cv2.threshold(gray_line, 200, 255, cv2.THRESH_BINARY_INV)

        ls = current_glyphs[0].metadata.get("shirorekha_y", int(h * 0.35))
        lb = current_glyphs[0].metadata.get("baseline_y", int(h * 0.75))

        old_spans = [(g.bbox[0], g.bbox[2]) for g in current_glyphs]
        curr_x1, curr_x2 = old_spans[glyph_idx]
        sub_w = curr_x2 - curr_x1

        # Calculate best split point near the ratio using body VPP valley
        mid_split = curr_x1 + int(sub_w * split_ratio)
        body_sub = bin_inv[ls + 2:lb, max(0, mid_split - 8):min(w, mid_split + 8)]
        if body_sub.size > 0:
            sub_vpp = np.sum(body_sub == 255, axis=0)
            best_offset = int(np.argmin(sub_vpp)) - 8
            mid_split = max(curr_x1 + 6, min(curr_x2 - 6, mid_split + best_offset))

        new_spans = []
        for i in range(len(current_glyphs)):
            if i == glyph_idx:
                new_spans.append((curr_x1, mid_split))
                new_spans.append((mid_split, curr_x2))
            else:
                new_spans.append(old_spans[i])

        return self._build_glyph_objects(gray_line, bin_inv, new_spans, line_id, ls, lb)

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

    def segment_document(self, preprocessed_img: np.ndarray, line_regions: List[Tuple[int, int]]) -> List[List[SegmentedGlyph]]:
        """
        Segments all extracted lines from a preprocessed document image.
        """
        all_document_glyphs = []
        for line_id, (sy, ey) in enumerate(line_regions):
            sy_clamped = max(0, sy)
            ey_clamped = min(preprocessed_img.shape[0], ey)
            line_crop = preprocessed_img[sy_clamped:ey_clamped, :]
            
            line_glyphs, _ = self.segment_line(line_crop, line_id=line_id)
            all_document_glyphs.append(line_glyphs)

        return all_document_glyphs
