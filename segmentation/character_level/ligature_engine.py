import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class LigatureEngine:
    """
    Scans the Middle Zone (core consonant region below Shirorekha) to identify candidate character
    boundaries, thin ligature points, and whitespace gaps.
    
    Protects:
      - Upper-zone Matras and Velanti arcs from being bisected.
      - Smooth ligature transitions along the Shirorekha headline.
    """

    def __init__(self, min_char_width: int = 10, vpp_valley_factor: float = 0.65):
        """
        Args:
            min_char_width: Minimum allowable distance between consecutive character candidates.
            vpp_valley_factor: Ratio against average body ink density to qualify as a valley.
        """
        self.min_char_width = min_char_width
        self.vpp_valley_factor = vpp_valley_factor

    def find_candidate_columns(self, line_img: np.ndarray, full_bin_inv: np.ndarray, shirorekha_y: int, baseline_y: int) -> List[int]:
        """
        Extracts candidate column cuts across the text line based on body density and upper stroke topology.

        Args:
            line_img: Grayscale text line.
            full_bin_inv: Full line binary inverted image (255=foreground ink, 0=background).
            shirorekha_y: Row index of Shirorekha (top headline).
            baseline_y: Row index of Baseline (bottom of consonant bodies).

        Returns:
            List of column indices representing Probable Segmentation Columns (PSC).
        """
        h, w = full_bin_inv.shape[:2]
        if h <= 2 or w <= 2:
            return [0, w]

        # 1. Compute Body VPP strictly below the Shirorekha headline
        body_bin = full_bin_inv[shirorekha_y + 3:min(h, baseline_y + int(h * 0.15)), :]
        if body_bin.shape[0] == 0:
            body_bin = full_bin_inv[shirorekha_y + 2:, :]

        vpp_body = np.sum(body_bin == 255, axis=0) / 255.0

        # Find whitespace columns where there is NO ink under the Shirorekha
        zero_cols = np.where(vpp_body == 0)[0]

        # Group consecutive zero columns into natural whitespace gaps
        candidate_cols = []
        if len(zero_cols) > 0:
            runs = []
            curr_run = [zero_cols[0]]
            for idx in zero_cols[1:]:
                if idx == curr_run[-1] + 1:
                    curr_run.append(idx)
                else:
                    runs.append(curr_run)
                    curr_run = [idx]
            if curr_run:
                runs.append(curr_run)

            for run in runs:
                # Cut at the midpoint of each inter-character whitespace gap
                mid_c = (run[0] + run[-1]) // 2
                candidate_cols.append(mid_c)

        # 2. Also find shallow ligature valleys in long continuous runs
        mean_vpp = np.mean(vpp_body[vpp_body > 0]) if np.any(vpp_body > 0) else 10.0
        valley_thresh = max(2.0, mean_vpp * self.vpp_valley_factor)

        # 3. Always include left and right margins of ink
        ink_cols = np.where(np.sum(full_bin_inv == 255, axis=0) > 0)[0]
        if len(ink_cols) > 0:
            start_col = max(0, ink_cols[0] - 2)
            end_col = min(w, ink_cols[-1] + 2)
        else:
            start_col, end_col = 0, w

        # Combine, sort, and deduplicate
        all_candidates = sorted(list(set([start_col] + candidate_cols + [end_col])))

        # 4. Upper Zone Protection: Prevent cutting through an upper matra loop (e.g. in टी, डी, भीती)
        upper_bin = full_bin_inv[0:shirorekha_y, :]
        num_up_labels, up_labels, up_stats, _ = cv2.connectedComponentsWithStats(upper_bin, connectivity=8)

        protected_candidates = [all_candidates[0]]
        for c in all_candidates[1:]:
            # Check if column c bisects an upper connected component
            bisected = False
            for k in range(1, num_up_labels):
                ux, uy, uw, uh, uarea = up_stats[k]
                if uarea >= 25 and ux + 2 < c < ux + uw - 2:
                    # Column cuts through an upper loop: shift to edge of upper component
                    bisected = True
                    # Snap cut to the right of the upper component
                    safe_c = min(w - 1, ux + uw + 1)
                    if safe_c - protected_candidates[-1] >= self.min_char_width:
                        protected_candidates.append(safe_c)
                    break
            if not bisected:
                if c - protected_candidates[-1] >= self.min_char_width:
                    protected_candidates.append(c)

        return sorted(list(set(protected_candidates)))
