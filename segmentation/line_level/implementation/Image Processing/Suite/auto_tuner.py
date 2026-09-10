import numpy as np
import itertools
from tqdm import tqdm

class AutoTuner:
    """
    Brute-force Auto-Tuner that mathematically evaluates segmentation quality
    to dynamically find the best parameters for a specific image.
    """
    def __init__(self, segmenter_class):
        self.segmenter_class = segmenter_class

    def score_segmentation(self, line_regions, binary_img, total_ink_pixels):
        """
        Calculates a mathematical 'Quality Score' using an Ink Coverage Heuristic.
        A good segmentation has:
        1. High % of document ink encapsulated inside the lines.
        2. Small bounding boxes (to prevent it from just wrapping the whole page).
        3. Uniform line heights (low standard deviation).
        """
        num_lines = len(line_regions)
        
        # Immediate rejection of garbage segmentations
        if num_lines < 3 or num_lines > 100:
            return -float('inf')
            
        heights = [end_y - start_y for start_y, end_y in line_regions]
        avg_height = np.mean(heights)
        
        # Reject if lines are ridiculously thin or thick
        if avg_height < 10 or avg_height > (binary_img.shape[0] / 3):
            return -float('inf')
            
        # Create a mask representing all extracted line regions
        lines_mask = np.zeros_like(binary_img)
        for start_y, end_y in line_regions:
            # Enforce valid bounds
            sy = max(0, start_y)
            ey = min(binary_img.shape[0], end_y)
            lines_mask[sy:ey, :] = 255
            
        # 1. Coverage Metric: How much ink is actually inside our lines?
        ink_inside = np.sum((binary_img == 255) & (lines_mask == 255))
        coverage = ink_inside / (total_ink_pixels + 1e-5)
        
        # 2. Area Metric: How much of the total page do our lines cover?
        total_line_area = np.sum(lines_mask == 255)
        total_image_area = binary_img.shape[0] * binary_img.shape[1]
        area_ratio = total_line_area / total_image_area
        
        # 3. Variance Metric
        height_std = np.std(heights)
        
        # SCORING FORMULA
        # We want maximum coverage (scale to 1000)
        # We penalize large area ratios (to force tight boxes)
        # We penalize height variance (to force uniform lines)
        score = (coverage * 1000.0) - (area_ratio * 400.0) - (height_std * 2.0)
        
        # Bonus for finding more lines, but capped so it doesn't encourage shattering
        score += min(num_lines * 5.0, 100.0)
        
        return score

    def tune(self, image_path, param_grid):
        """
        Runs a grid search over the parameter combinations and returns the best parameters.
        """
        # Create all possible combinations of parameters
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        segmenter = self.segmenter_class()
        
        best_score = -float('inf')
        best_params = None
        best_regions = None
        
        print(f"[*] Auto-Tuning over {len(combinations)} parameter combinations...")
        
        # Pre-calculate base binary image and ink metrics to save time
        _, base_binary = segmenter.preprocess(image_path)
        total_ink_pixels = np.sum(base_binary == 255)
        
        if total_ink_pixels == 0:
            print("[!] Document has no ink. Auto-Tuner aborting.")
            return None, 0, None

        for params in tqdm(combinations, desc="Auto-Tuning"):
            try:
                # Run segmentation (suppress debug output)
                _, _, line_regions, _ = segmenter.segment(image_path, **params)
                
                # Evaluate using Ink Coverage
                score = self.score_segmentation(line_regions, base_binary, total_ink_pixels)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_regions = line_regions
            except Exception as e:
                # If a combination crashes (e.g. invalid kernel size), skip it
                continue
                
        return best_params, best_score, best_regions
