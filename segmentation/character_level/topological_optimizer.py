import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class TopologicalSyllableOptimizer:
    """
    Mathematical Dynamic Programming (Viterbi-style) & Topological Stroke Graph Optimizer
    for Modi Script Character & Syllable Segmentation.
    
    Synthesizes:
      - Option 3: Topological Stroke Analysis (detects loops, consonant anchors, curved/straight kanas).
      - Option 2: Dynamic Programming Shortest-Path Energy Optimization over candidate cut DAG.
    """

    def __init__(
        self,
        min_char_width: int = 14,
        ideal_char_width: int = 34,
        max_char_width: int = 65,
        orphan_penalty: float = 1000.0
    ):
        self.min_char_width = min_char_width
        self.ideal_char_width = ideal_char_width
        self.max_char_width = max_char_width
        self.orphan_penalty = orphan_penalty

    def _analyze_stroke_topology(self, slice_bin: np.ndarray, shirorekha_y: int) -> Dict[str, Any]:
        """
        Analyzes the mathematical and topological morphology of an ink slice under the Shirorekha.
        Detects:
          - has_loop: Closed topological loop (e.g. circle of 'ग', 'श', 'ळ', 'म', 'क')
          - is_trailing_stem: Narrow vertical bar or curved hook kana (ा)
          - is_orphan_noise: Negligible ink fragment
        """
        h, w = slice_bin.shape[:2]
        if w <= 0 or h <= 0:
            return {"is_empty": True, "has_loop": False, "is_stem": False, "ink_area": 0}

        body_crop = slice_bin[shirorekha_y + 2:, :]
        if body_crop.size == 0 or body_crop.shape[0] < 4 or body_crop.shape[1] < 2:
            return {"is_empty": True, "has_loop": False, "is_stem": False, "ink_area": 0}

        ink_pixels = int(np.sum(body_crop == 255))
        if ink_pixels < 15:
            return {"is_empty": True, "has_loop": False, "is_stem": False, "ink_area": ink_pixels}

        # 1. Topological Loop Detection (Euler number / Contours with holes)
        # Invert body crop: Background holes will be enclosed by ink
        contours, hierarchy = cv2.findContours(body_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        has_loop = False
        if hierarchy is not None and len(hierarchy[0]) > 1:
            # Check if any contour has internal children (enclosed loop hole)
            for h_entry in hierarchy[0]:
                if h_entry[2] != -1 or h_entry[3] != -1:
                    has_loop = True
                    break

        # 2. Kana / Stem Detection (Straight or curved vertical hook dropping from Shirorekha)
        col_sums = np.sum(body_crop == 255, axis=0) / 255.0
        max_col_height = float(np.max(col_sums)) if len(col_sums) > 0 else 0
        body_h = body_crop.shape[0]

        is_stem = False
        if not has_loop and w <= 20:
            # If height of ink stroke is substantial (drops at least 40% of body height)
            if max_col_height >= body_h * 0.35:
                is_stem = True

        return {
            "is_empty": False,
            "has_loop": has_loop,
            "is_stem": is_stem,
            "ink_area": ink_pixels,
            "width": w
        }

    def compute_segment_cost(
        self,
        full_bin_inv: np.ndarray,
        x_start: int,
        x_end: int,
        shirorekha_y: int,
        baseline_y: int,
        vpp_body: np.ndarray
    ) -> float:
        """
        Computes the mathematical Energy / Penalty Cost E(x_start, x_end) for a candidate character span.
        Lower cost = Higher probability of being a complete, valid Modi Akshara.
        """
        w_seg = x_end - x_start
        if w_seg <= 0:
            return float('inf')

        # 1. Width penalty (Gaussian-like penalty around ideal character width)
        if w_seg < self.min_char_width:
            cost_width = ((self.min_char_width - w_seg) ** 2) * 5.0
        elif w_seg > self.max_char_width:
            cost_width = ((w_seg - self.max_char_width) ** 2) * 3.0
        else:
            cost_width = abs(w_seg - self.ideal_char_width) * 0.4

        # 2. Cut Valley Cost at Boundaries x_start and x_end
        # True character boundaries must occur at low ink density (valleys on Shirorekha bridge)
        v_left = float(vpp_body[min(x_start, len(vpp_body) - 1)]) if len(vpp_body) > 0 else 0
        v_right = float(vpp_body[min(max(0, x_end - 1), len(vpp_body) - 1)]) if len(vpp_body) > 0 else 0
        cost_cut = (v_left + v_right) * 2.5

        # 3. Topological Morphology Analysis
        slice_crop = full_bin_inv[:, x_start:x_end]
        topo = self._analyze_stroke_topology(slice_crop, shirorekha_y)

        cost_topo = 0.0
        if topo["is_empty"]:
            cost_topo = self.orphan_penalty * 2.0
        elif topo["is_stem"]:
            # An isolated vertical stem or orphan kana has a HUGE penalty if treated as an independent character
            cost_topo = self.orphan_penalty
        elif topo["has_loop"]:
            # Rewards well-formed consonant base loops
            cost_topo = -15.0

        total_energy = cost_width + cost_cut + cost_topo
        return total_energy

    def optimize_line_segmentation(
        self,
        full_bin_inv: np.ndarray,
        primitive_cuts: List[int],
        shirorekha_y: int,
        baseline_y: int
    ) -> List[Tuple[int, int]]:
        """
        Executes Dynamic Programming (Viterbi Shortest Path) over the Directed Acyclic Graph (DAG)
        of candidate cut points to find the globally optimal character boundary sequence.
        """
        if len(primitive_cuts) < 2:
            return []

        h, w = full_bin_inv.shape[:2]
        body_bin = full_bin_inv[shirorekha_y + 2:, :]
        vpp_body = np.sum(body_bin == 255, axis=0) / 255.0 if body_bin.size > 0 else np.zeros(w)

        N = len(primitive_cuts)
        dp = [float('inf')] * N
        parent = [-1] * N
        dp[0] = 0.0

        # Maximum lookahead: A single character span can combine at most 4 consecutive primitives (e.g. loop + stem + kana)
        max_jump = 5

        for j in range(1, N):
            for i in range(max(0, j - max_jump), j):
                x_i = primitive_cuts[i]
                x_j = primitive_cuts[j]
                
                # Compute transition energy
                step_cost = self.compute_segment_cost(
                    full_bin_inv, x_i, x_j, shirorekha_y, baseline_y, vpp_body
                )
                
                if dp[i] + step_cost < dp[j]:
                    dp[j] = dp[i] + step_cost
                    parent[j] = i

        # Backtrack optimal sequence of cuts
        optimal_indices = []
        curr = N - 1
        while curr != -1:
            optimal_indices.append(curr)
            curr = parent[curr]
        optimal_indices.reverse()

        # Build finalized character spans
        optimal_spans = []
        for k in range(len(optimal_indices) - 1):
            i_idx = optimal_indices[k]
            j_idx = optimal_indices[k + 1]
            x1 = primitive_cuts[i_idx]
            x2 = primitive_cuts[j_idx]

            # Tighten horizontal boundaries to foreground ink
            slice_crop = full_bin_inv[:, x1:x2]
            col_sums = np.sum(slice_crop == 255, axis=0)
            non_zeros = np.where(col_sums > 0)[0]
            if len(non_zeros) > 0 and np.sum(slice_crop == 255) >= 20:
                tight_x1 = max(0, x1 + non_zeros[0] - 2)
                tight_x2 = min(w, x1 + non_zeros[-1] + 3)
                if tight_x2 - tight_x1 >= 8:
                    optimal_spans.append((tight_x1, tight_x2))

        return optimal_spans
