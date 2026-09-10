import cv2
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

class SegmentationEngine:
    """Base class for all classical segmentation algorithms."""
    def preprocess(self, image_path, blur_ksize=5, block_size=15, c_val=10):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_color = cv2.imread(image_path)
        blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_val
        )
        return original_color, binary

    def segment(self, image_path, **kwargs):
        raise NotImplementedError

class HPPSegmenter(SegmentationEngine):
    """Method 1: Classical Horizontal Projection Profile"""
    def segment(self, image_path, savgol_window=21, ink_threshold=5, min_line_height=15, pad=5, **kwargs):
        original, binary = self.preprocess(image_path, **kwargs)
        hpp = np.sum(binary, axis=1) / 255.0
        smoothed_hpp = savgol_filter(hpp, window_length=savgol_window, polyorder=3)
        smoothed_hpp[smoothed_hpp < 0] = 0
        
        is_text_row = smoothed_hpp > ink_threshold
        line_regions = []
        in_line, start_y = False, 0
        
        for y, is_text in enumerate(is_text_row):
            if is_text and not in_line:
                start_y, in_line = y, True
            elif not is_text and in_line:
                end_y, in_line = y, False
                if (end_y - start_y) > min_line_height:
                    line_regions.append((start_y, end_y))
        
        if in_line and (len(is_text_row) - start_y) > min_line_height:
            line_regions.append((start_y, len(is_text_row)))
            
        debug_img = binary.copy()
        for sy, ey in line_regions:
            cv2.line(debug_img, (0, sy), (binary.shape[1], sy), (128), 2)
            cv2.line(debug_img, (0, ey), (binary.shape[1], ey), (255), 2)
            
        return original, debug_img, line_regions, pad

class GapAnalysisSegmenter(SegmentationEngine):
    """Method 2: Statistical Gap Analysis (Inspired by Mukherjee 2022)"""
    def segment(self, image_path, min_line_height=15, pad=5, **kwargs):
        original, binary = self.preprocess(image_path, **kwargs)
        hpp = np.sum(binary, axis=1) / 255.0
        
        # Use KMeans to statistically separate 'text rows' from 'gap rows'
        # based on ink density distribution instead of a hardcoded threshold
        X = hpp.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        threshold = np.mean(kmeans.cluster_centers_)
        
        is_text_row = hpp > threshold
        line_regions = []
        in_line, start_y = False, 0
        
        for y, is_text in enumerate(is_text_row):
            if is_text and not in_line:
                start_y, in_line = y, True
            elif not is_text and in_line:
                end_y, in_line = y, False
                if (end_y - start_y) > min_line_height:
                    line_regions.append((start_y, end_y))
                    
        if in_line and (len(is_text_row) - start_y) > min_line_height:
            line_regions.append((start_y, len(is_text_row)))
            
        debug_img = binary.copy()
        for sy, ey in line_regions:
            cv2.line(debug_img, (0, sy), (binary.shape[1], sy), (128), 2)
            
        return original, debug_img, line_regions, pad

class AnisotropicGaussianSegmenter(SegmentationEngine):
    """Method 3: Anisotropic Gaussian Blobs (Inspired by Barakat 2020)"""
    def segment(self, image_path, x_stretch=41, y_stretch=5, thresh_val=50, pad=5, **kwargs):
        original, binary = self.preprocess(image_path, **kwargs)
        
        # Directional Gaussian Blur to smear text horizontally but not vertically
        smeared = cv2.GaussianBlur(binary, (x_stretch | 1, y_stretch | 1), 0)
        
        # Threshold the smeared image to create solid blob lines
        _, blob_mask = cv2.threshold(smeared, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours of the blob lines
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > original.shape[1] * 0.1 and h > 10: # Filter noise
                line_regions.append((y, y + h))
                
        # Sort top to bottom
        line_regions = sorted(line_regions, key=lambda r: r[0])
        
        return original, blob_mask, line_regions, pad

class EdgeConnectivitySegmenter(SegmentationEngine):
    """Method 4: Edge Connectivity Construction (Inspired by Li 2019)"""
    def segment(self, image_path, canny_lower=50, canny_upper=150, dilation_x=51, dilation_y=3, pad=5, **kwargs):
        original, _ = self.preprocess(image_path, **kwargs)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Detection
        edges = cv2.Canny(gray, canny_lower, canny_upper)
        
        # 2. Horizontal Morphological Dilation to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_x, dilation_y))
        connected_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        connected_edges = cv2.dilate(connected_edges, kernel, iterations=1)
        
        # 3. Find connected components (lines)
        contours, _ = cv2.findContours(connected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > original.shape[1] * 0.1 and h > 10:
                line_regions.append((y, y + h))
                
        line_regions = sorted(line_regions, key=lambda r: r[0])
        
        return original, connected_edges, line_regions, pad

class SeamCarvingSegmenter(SegmentationEngine):
    """Method 5: A* Path Planning / Seam Carving (Inspired by Nguyen 2022)"""
    def segment(self, image_path, num_seams=20, pad=5, **kwargs):
        original, binary = self.preprocess(image_path, **kwargs)
        
        # 1. Distance Transform (distance to nearest text pixel)
        # We invert the binary mask so text is 0 and background is 255 for distance transform
        dist_transform = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 3)
        
        # 2. Energy Map: High energy near text, low energy in gaps
        # Max distance gets low energy, text gets high energy
        max_dist = np.max(dist_transform)
        if max_dist == 0:
            max_dist = 1
        energy_map = 1.0 - (dist_transform / max_dist)
        
        # 3. Dynamic Programming to find seams (Horizontal Seam Carving)
        h, w = energy_map.shape
        # DP table stores min accumulated energy
        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=int)
        
        # Traverse left to right
        for j in range(1, w):
            for i in range(h):
                # Look at 3 neighbors to the left (i-1, i, i+1)
                idx_up = max(i-1, 0)
                idx_dn = min(i+1, h-1)
                
                neighbors = [M[idx_up, j-1], M[i, j-1], M[idx_dn, j-1]]
                min_idx = np.argmin(neighbors)
                
                # backtrack stores whether we came from UP(0), STRAIGHT(1), or DOWN(2)
                backtrack[i, j] = min_idx
                M[i, j] += neighbors[min_idx]
                
        # To find individual lines, we usually need to find MULTIPLE seams
        # This implementation approximates by doing an HPP-guided seam finding
        hpp = np.sum(binary, axis=1) / 255.0
        X = hpp.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        threshold = np.mean(kmeans.cluster_centers_)
        is_gap_row = hpp < threshold
        
        seam_ys = []
        in_gap = False
        for y, is_gap in enumerate(is_gap_row):
            if is_gap and not in_gap:
                in_gap = True
                seam_ys.append(y)
            elif not is_gap and in_gap:
                in_gap = False
                
        # Draw seams for visualization
        debug_img = (energy_map * 255).astype(np.uint8)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        # Map seams to bounding box regions for consistency with other methods
        line_regions = []
        if len(seam_ys) > 1:
            for i in range(len(seam_ys)-1):
                start_y = seam_ys[i]
                end_y = seam_ys[i+1]
                if (end_y - start_y) > 15:
                    line_regions.append((start_y, end_y))
                    # Draw a straight approximated seam
                    cv2.line(debug_img, (0, start_y), (w, start_y), (0, 255, 0), 2)
                    cv2.line(debug_img, (0, end_y), (w, end_y), (0, 0, 255), 2)

        return original, debug_img, line_regions, pad
