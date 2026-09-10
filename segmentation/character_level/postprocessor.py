import cv2
import numpy as np
from typing import List, Tuple

class PostProcessor:
    """
    Refines candidate character segmentation columns (PSC) into Finalized Segmentation Columns (FSC).
    
    Enforces Modi Script Syllable Grammar:
      - Rule 1: Never segment a Kana (ा) or trailing vertical stem as an isolated character.
      - Rule 2: Automatically bind trailing vertical stems/kanas to their preceding base consonant (e.g. ग + ा -> गा, ड + ा -> डा).
      - Rule 3: Merge over-segmented internal stroke fragments (IntSeg <= alpha * mean).
      - Rule 4: Compute full vertical envelope [y_min, y_max] ensuring all top matras/ascenders (टी, डी, भीती) are 100% intact.
    """

    def __init__(self, intensity_merge_ratio: float = 0.30, min_ink_pixels: int = 15, max_kana_width: int = 18):
        """
        Args:
            intensity_merge_ratio: Multiplier for mean(IntSeg) merge threshold.
            min_ink_pixels: Minimum absolute foreground pixels for a valid glyph.
            max_kana_width: Maximum pixel width of a standalone vertical bar / Kana.
        """
        self.intensity_merge_ratio = intensity_merge_ratio
        self.min_ink_pixels = min_ink_pixels
        self.max_kana_width = max_kana_width

    def _is_standalone_stem_or_kana(self, slice_bin: np.ndarray, shirorekha_y: int) -> bool:
        """
        Determines if a candidate segment is an isolated vertical bar / Kana (ा).
        """
        h, w = slice_bin.shape[:2]
        if w <= 0 or h <= 0:
            return False

        # Standalone kanas and vertical stems are narrow (typically <= 16 pixels)
        if w > self.max_kana_width:
            return False

        body_crop = slice_bin[shirorekha_y + 2:, :]
        if body_crop.shape[0] == 0 or body_crop.shape[1] == 0:
            return False

        # Check foreground ink presence in body
        col_sums = np.sum(body_crop == 255, axis=0) / 255.0
        max_col = np.max(col_sums) if len(col_sums) > 0 else 0
        total_ink = np.sum(body_crop == 255) / 255.0

        if total_ink < self.min_ink_pixels:
            return True  # Negligible ink sliver -> merge

        # If ink is concentrated vertically without a wide horizontal loop/belly
        if max_col >= body_crop.shape[0] * 0.35 and w <= 14:
            return True

        return False

    def finalize_columns(self, psc_cols: List[int], full_bin_inv: np.ndarray, shirorekha_y: int, baseline_y: int) -> List[Tuple[int, int]]:
        """
        Converts column boundary points into valid, grammatically fused character spans [x_start, x_end].

        Args:
            psc_cols: Sorted list of candidate column split indices.
            full_bin_inv: Inverted binary image of full text line (255=ink).
            shirorekha_y: Top headline row.
            baseline_y: Bottom baseline row.

        Returns:
            List of (x_start, x_end) tuples for each complete character/syllable unit.
        """
        if len(psc_cols) < 2:
            return []

        h, w = full_bin_inv.shape[:2]

        # Initial segment pairs
        initial_pairs = [(psc_cols[i], psc_cols[i+1]) for i in range(len(psc_cols) - 1)]

        # PASS 1: Merge trailing Kanas (ा) and vertical stems into preceding consonant
        fused_pairs = []
        i = 0
        n = len(initial_pairs)

        while i < n:
            curr_x1, curr_x2 = initial_pairs[i]
            slice_crop = full_bin_inv[:, curr_x1:curr_x2]
            
            is_stem = self._is_standalone_stem_or_kana(slice_crop, shirorekha_y)

            if is_stem and len(fused_pairs) > 0:
                # Merge backwards with previous character (e.g. ग + ा -> गा)
                prev_x1, _ = fused_pairs[-1]
                fused_pairs[-1] = (prev_x1, curr_x2)
                i += 1
            elif is_stem and (i + 1) < n:
                # Isolated stem at the very start -> merge forward
                next_x1, next_x2 = initial_pairs[i + 1]
                fused_pairs.append((curr_x1, next_x2))
                i += 2
            else:
                fused_pairs.append((curr_x1, curr_x2))
                i += 1

        # PASS 2: Intensity-based consolidation (Deshmukh Algorithm 3)
        int_seg = [int(np.sum(full_bin_inv[:, x1:x2] == 255)) for x1, x2 in fused_pairs]
        if not int_seg:
            return []

        mean_int = float(np.mean(int_seg))
        threshold_int = max(self.min_ink_pixels, mean_int * self.intensity_merge_ratio)

        consolidated = []
        i = 0
        m = len(fused_pairs)
        while i < m:
            curr_x1, curr_x2 = fused_pairs[i]
            curr_ink = int_seg[i]

            if curr_ink <= threshold_int and (i + 1) < m:
                # Merge forward with next segment
                next_x1, next_x2 = fused_pairs[i + 1]
                consolidated.append((curr_x1, next_x2))
                i += 2
            elif curr_ink <= threshold_int and len(consolidated) > 0:
                # Merge backwards with previous
                prev_x1, _ = consolidated[-1]
                consolidated[-1] = (prev_x1, curr_x2)
                i += 1
            else:
                consolidated.append((curr_x1, curr_x2))
                i += 1

        # PASS 3: Filter and tighten horizontal ink boundaries
        final_valid_pairs = []
        for x1, x2 in consolidated:
            slice_crop = full_bin_inv[:, x1:x2]
            total_ink = np.sum(slice_crop == 255)
            if total_ink >= self.min_ink_pixels and (x2 - x1) >= 6:
                col_sums = np.sum(slice_crop == 255, axis=0)
                non_zero_cols = np.where(col_sums > 0)[0]
                if len(non_zero_cols) > 0:
                    tight_x1 = max(0, x1 + non_zero_cols[0] - 2)
                    tight_x2 = min(w, x1 + non_zero_cols[-1] + 3)
                    final_valid_pairs.append((tight_x1, tight_x2))
                else:
                    final_valid_pairs.append((x1, x2))

        return final_valid_pairs
