import cv2
import numpy as np

class FFTAnalyzer:
    """
    Mathematical analyzer that uses Fast Fourier Transform (FFT) to convert 
    an image into the frequency domain and instantly calculate the 
    average pixel gap between lines.
    """
    @staticmethod
    def calculate_optimal_parameters(image_path):
        # 1. Load and binarize
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # 2. Get 1D Spatial Signal (Horizontal Projection Profile)
        hpp = np.sum(binary, axis=1) / 255.0
        N = len(hpp)
        
        # 3. Autocorrelation Function (ACF)
        # Subtract the mean to remove the DC offset
        hpp_centered = hpp - np.mean(hpp)
        
        # Compute full autocorrelation
        acf = np.correlate(hpp_centered, hpp_centered, mode='full')
        
        # The result is symmetric, so we only take the second half (lags >= 0)
        acf = acf[len(acf)//2:]
        
        # 4. Find the Dominant Spatial Period using Peak Detection
        # We look for the first major peak after lag=0
        from scipy.signal import find_peaks
        
        # Prominence ensures we find a distinct periodic peak, distance ensures it's not immediately next to 0
        peaks, properties = find_peaks(acf, prominence=0.05 * np.max(acf), distance=15)
        
        # 5. Calculate Pixel Distance
        if len(peaks) == 0:
            print("[!] ACF failed to find a periodic peak. Falling back to default 50px.")
            pixel_period = 50 # Fallback
        else:
            # The lag distance to the first major peak is the exact average pixel period
            pixel_period = float(peaks[0])
            
        # 6. Map Pixel Period to Algorithm Parameters
        # y_stretch: small fraction of period — enough to smear lines without bleeding into adjacent ones
        optimal_y_stretch = max(3, int(pixel_period / 6.0))
        if optimal_y_stretch % 2 == 0:
            optimal_y_stretch += 1
        
        # x_stretch: should span roughly the width needed to connect word-level glyphs
        # We use 1.5x the period as a heuristic for horizontal smear
        optimal_x_stretch = max(21, int(pixel_period * 1.5))
        if optimal_x_stretch % 2 == 0:
            optimal_x_stretch += 1
        
        # savgol_window: smoothing window ~ half the period
        optimal_savgol = max(5, int(pixel_period / 2.0))
        if optimal_savgol % 2 == 0:
            optimal_savgol += 1
        optimal_savgol = min(51, optimal_savgol)
        
        # ink_threshold: derived from the HPP signal's non-zero median
        non_zero_hpp = hpp[hpp > 0]
        optimal_ink_threshold = max(2, int(np.median(non_zero_hpp) * 0.3)) if len(non_zero_hpp) > 0 else 5
        
        return {
            "spatial_period_px": pixel_period,
            "y_stretch":         optimal_y_stretch,
            "x_stretch":         optimal_x_stretch,
            "savgol_window":     optimal_savgol,
            "ink_threshold":     optimal_ink_threshold,
        }
