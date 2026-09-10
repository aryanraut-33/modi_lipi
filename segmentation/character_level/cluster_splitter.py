import cv2
import numpy as np
from typing import List, Tuple

class ClusterSplitter:
    """
    Detects and splits touching and overlapping Modi character clusters.
    
    References:
      - Deshmukh & Kolhe (2019): A Modified Approach for the Segmentation of 
        Unconstrained Cursive Modi Touching Characters Cluster.
      - Deshmukh & Kolhe (2024): Overlapping & Touching Character Cluster Disambiguation.
    """

    def __init__(self, cluster_width_multiplier: float = 1.45, min_split_width: int = 12):
        """
        Args:
            cluster_width_multiplier: Ratio above median width to classify a segment as a cluster.
            min_split_width: Minimum pixel width of a child segment after splitting.
        """
        self.cluster_width_multiplier = cluster_width_multiplier
        self.min_split_width = min_split_width

    def evaluate_and_split_clusters(self, candidate_cols: List[int], middle_zone: np.ndarray, full_bin_inv: np.ndarray) -> List[int]:
        """
        Checks candidate intervals [c_i, c_{i+1}] and splits any oversized touching clusters.

        Args:
            candidate_cols: Initial list of column cuts.
            middle_zone: 2D array of middle zone.
            full_bin_inv: Full line binary inverted image (255=ink).

        Returns:
            Refined list of column boundaries including split points.
        """
        if len(candidate_cols) < 2:
            return candidate_cols

        widths = [candidate_cols[i+1] - candidate_cols[i] for i in range(len(candidate_cols) - 1)]
        if not widths:
            return candidate_cols

        median_w = float(np.median(widths))
        threshold_w = max(self.min_split_width * 2, median_w * self.cluster_width_multiplier)

        refined_cols = [candidate_cols[0]]

        for i in range(len(candidate_cols) - 1):
            c_start = candidate_cols[i]
            c_end = candidate_cols[i+1]
            seg_w = c_end - c_start

            if seg_w > threshold_w:
                # Segment is abnormally wide: Perform touching cluster dissection
                split_points = self._split_cluster(c_start, c_end, middle_zone, full_bin_inv)
                for sp in split_points:
                    if sp > refined_cols[-1] + self.min_split_width and sp < c_end - self.min_split_width:
                        refined_cols.append(sp)

            refined_cols.append(c_end)

        return sorted(list(set(refined_cols)))

    def _split_cluster(self, c_start: int, c_end: int, middle_zone: np.ndarray, full_bin_inv: np.ndarray) -> List[int]:
        """
        Applies Connected Component analysis and VPP valley searching within [c_start, c_end].
        """
        sub_mid = middle_zone[:, c_start:c_end]
        h_mid, w_sub = sub_mid.shape[:2]
        if w_sub < self.min_split_width * 2:
            return []

        # Convert to binary
        if len(sub_mid.shape) == 3:
            sub_gray = cv2.cvtColor(sub_mid, cv2.COLOR_BGR2GRAY)
        else:
            sub_gray = sub_mid.copy()
        _, sub_bin = cv2.threshold(sub_gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 1. Connected Component overlap ratio in lower half of middle zone
        lower_half = sub_bin[h_mid // 2:, :]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lower_half, connectivity=8)

        # If there are multiple distinct components in lower-half, find their dividing gap
        valid_comps = [k for k in range(1, num_labels) if stats[k, cv2.CC_STAT_AREA] >= 15]
        if len(valid_comps) >= 2:
            # Sort components left to right
            valid_comps = sorted(valid_comps, key=lambda k: stats[k, cv2.CC_STAT_LEFT])
            split_points = []
            for j in range(len(valid_comps) - 1):
                c1_right = stats[valid_comps[j], cv2.CC_STAT_LEFT] + stats[valid_comps[j], cv2.CC_STAT_WIDTH]
                c2_left = stats[valid_comps[j+1], cv2.CC_STAT_LEFT]
                
                if c2_left >= c1_right:
                    # Non-overlapping horizontal gap
                    split_x = c_start + (c1_right + c2_left) // 2
                    split_points.append(split_x)
                else:
                    # Partial overlap: Find minimum column density between centroids
                    cent_x1 = int(centroids[valid_comps[j]][0])
                    cent_x2 = int(centroids[valid_comps[j+1]][0])
                    if cent_x2 > cent_x1:
                        sub_vpp = np.sum(sub_bin[:, cent_x1:cent_x2] == 255, axis=0)
                        if len(sub_vpp) > 0:
                            min_local = np.argmin(sub_vpp)
                            split_points.append(c_start + cent_x1 + int(min_local))
            if split_points:
                return split_points

        # 2. Entirely Touching Cluster: VPP Valley Analysis
        sub_vpp = np.sum(sub_bin == 255, axis=0)
        # Search for valleys in the middle 60% of the cluster
        margin = int(w_sub * 0.20)
        search_region = sub_vpp[margin:w_sub - margin]
        if len(search_region) > 0:
            min_val = np.min(search_region)
            mean_val = np.mean(sub_vpp)
            # If there's a distinct dip below mean density
            if min_val < mean_val * 0.8:
                valley_idx = margin + int(np.argmin(search_region))
                return [c_start + valley_idx]

        # Fallback: Equal midpoint split if cluster is exceptionally wide (>= 2.2x median)
        return [c_start + w_sub // 2]
