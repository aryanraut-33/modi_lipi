import cv2
import numpy as np
from typing import Tuple, Dict, Any

class ZonalAnalyzer:
    """
    Analyzes the horizontal projection profile of a Modi text-line image to
    accurately locate the Shirorekha (headline L_s) and the Baseline (L_b),
    partitioning the line into Upper (Matras/Ascenders), Middle (Consonant Bodies),
    and Lower (Descenders/Subscripts) structural zones.
    
    References:
      - Deshmukh & Kolhe (2019, 2024): Horizontal Projection Profile Maxima/Minima for Treg.
    """

    def __init__(self, upper_search_ratio: float = 0.50, lower_search_ratio: float = 0.65):
        """
        Args:
            upper_search_ratio: Upper fraction of line height within which Shirorekha must lie.
            lower_search_ratio: Fraction of line height below which Baseline must lie.
        """
        self.upper_search_ratio = upper_search_ratio
        self.lower_search_ratio = lower_search_ratio

    def analyze_line(self, line_img: np.ndarray) -> Dict[str, Any]:
        """
        Performs 3-zone topological decomposition on a grayscale or binary text-line image.

        Args:
            line_img: 2D numpy array (grayscale / binary text line, 0=black ink, 255=white paper).

        Returns:
            Dictionary containing:
              - 'shirorekha_y': int (L_s row index, top headline)
              - 'baseline_y': int (L_b row index, bottom baseline)
              - 'upper_zone': np.ndarray crop [0:L_s, :]
              - 'middle_zone': np.ndarray crop [L_s:L_b, :]
              - 'lower_zone': np.ndarray crop [L_b:H, :]
              - 'hpp': 1D array of horizontal projection values
        """
        h, w = line_img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Input line image is empty.")

        # Ensure inverted binary representation (255 = foreground ink, 0 = background)
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 1. Compute Horizontal Projection Profile (HPP)
        hpp = np.sum(bin_inv == 255, axis=1)

        # Smooth HPP with a 1D Gaussian kernel to avoid picking single-pixel noise spikes
        kernel_size = max(3, (h // 12) | 1)
        kernel = cv2.getGaussianKernel(kernel_size, sigma=1.5)
        smoothed_hpp = cv2.filter2D(hpp.astype(np.float32), -1, kernel).flatten()

        # 2. Detect Shirorekha (L_s): Maximum peak in top upper search band (e.g. top 50%)
        search_top_limit = max(4, int(h * self.upper_search_ratio))
        # Ensure we skip first 2 edge pixels to avoid border artifacts
        start_y = min(2, search_top_limit - 2)
        shirorekha_y = int(np.argmax(smoothed_hpp[start_y:search_top_limit])) + start_y

        # Fallback if Shirorekha detection is degenerate
        if shirorekha_y <= 1 or shirorekha_y >= search_top_limit:
            shirorekha_y = max(2, int(h * 0.30))

        # 3. Detect Baseline (L_b): The bottom boundary where core consonant bodies end
        search_bottom_start = max(shirorekha_y + int(h * 0.25), int(h * self.lower_search_ratio))
        search_bottom_start = min(search_bottom_start, h - 3)

        if search_bottom_start < h - 2:
            baseline_y = search_bottom_start + int(np.argmin(smoothed_hpp[search_bottom_start:]))
        else:
            baseline_y = int(h * 0.85)

        # Guarantee strictly L_s < L_b with meaningful middle-band gap
        min_mid_height = max(10, int(h * 0.35))
        if baseline_y <= shirorekha_y + min_mid_height:
            baseline_y = min(h - 1, shirorekha_y + min_mid_height)

        # 4. Partition into 3 Zones
        upper_zone = line_img[0:shirorekha_y, :]
        middle_zone = line_img[shirorekha_y:baseline_y, :]
        lower_zone = line_img[baseline_y:h, :]

        return {
            "shirorekha_y": shirorekha_y,
            "baseline_y": baseline_y,
            "upper_zone": upper_zone,
            "middle_zone": middle_zone,
            "lower_zone": lower_zone,
            "hpp": hpp,
            "smoothed_hpp": smoothed_hpp,
            "height": h,
            "width": w
        }
