import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

class HPPSegmenter:
    """
    Classical Computer Vision Line Segmenter using Horizontal Projection Profile (HPP).
    Requires ZERO model training.
    """
    def __init__(self, output_dir="segmented_lines", debug=False,
                 blur_ksize=5, block_size=15, c_val=10, 
                 savgol_window=21, ink_threshold=5, min_line_height=15, pad=5):
        self.output_dir = output_dir
        self.debug = debug
        self.blur_ksize = blur_ksize
        self.block_size = block_size
        self.c_val = c_val
        self.savgol_window = savgol_window
        self.ink_threshold = ink_threshold
        self.min_line_height = min_line_height
        self.pad = pad
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess(self, image_path):
        """Loads and binarizes the image (Text = White, Background = Black)."""
        # 1. Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Keep original for cropping later
        original_color = cv2.imread(image_path)

        # 2. Gaussian Blur to remove high-frequency noise
        blurred = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)

        # 3. Adaptive Thresholding (Handles uneven lighting in historical docs)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.block_size, self.c_val
        )
        
        # 4. Optional: Morphological Opening to remove tiny speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return original_color, clean_binary

    def compute_hpp(self, binary_image):
        """Calculates the Horizontal Projection Profile."""
        # Sum all pixel values across the horizontal axis (axis=1)
        # Divide by 255 so the value represents the *number of text pixels* per row
        hpp = np.sum(binary_image, axis=1) / 255.0
        return hpp

    def segment_lines(self, image_path):
        """Executes the HPP pipeline and crops the text lines."""
        print(f"[*] Processing: {os.path.basename(image_path)}")
        
        original, binary = self.preprocess(image_path)
        hpp = self.compute_hpp(binary)

        # Smooth the histogram to prevent accidental splits inside letters with gaps (like 'i' or 'j')
        # savgol_filter applies a polynomial smoothing that preserves peak heights better than moving average
        smoothed_hpp = savgol_filter(hpp, window_length=self.savgol_window, polyorder=3)
        smoothed_hpp[smoothed_hpp < 0] = 0 # Remove negative artifacts from smoothing

        # --- Detect Text Lines ---
        # We find contiguous blocks of rows where the number of text pixels exceeds a threshold.
        # Threshold: E.g., at least 5 pixels of ink must exist on a row to consider it text.
        is_text_row = smoothed_hpp > self.ink_threshold
        
        line_regions = []
        in_line = False
        start_y = 0

        for y, is_text in enumerate(is_text_row):
            if is_text and not in_line:
                # We just entered a text line
                start_y = y
                in_line = True
            elif not is_text and in_line:
                # We just exited a text line (hit a valley)
                end_y = y
                in_line = False
                
                # Filter out garbage: A valid Modi script line must be at least 15 pixels tall
                if (end_y - start_y) > self.min_line_height:
                    line_regions.append((start_y, end_y))

        # Handle edge case where text hits the absolute bottom of the image
        if in_line and (len(is_text_row) - start_y) > self.min_line_height:
            line_regions.append((start_y, len(is_text_row)))

        # --- Visual Debugging (Optional) ---
        if self.debug:
            self._plot_debug(original, binary, smoothed_hpp, line_regions)

        # --- Crop and Save ---
        saved_paths = []
        for idx, (start_y, end_y) in enumerate(line_regions):
            # Add a small 5-pixel padding top and bottom so text isn't cut off tightly
            y1 = max(0, start_y - self.pad)
            y2 = min(original.shape[0], end_y + self.pad)
            
            line_crop = original[y1:y2, :]
            
            out_path = os.path.join(self.output_dir, f"line_{idx:03d}.jpg")
            cv2.imwrite(out_path, line_crop)
            saved_paths.append(out_path)
            
        print(f"[+] Successfully extracted {len(line_regions)} lines to '{self.output_dir}'")
        return saved_paths

    def _plot_debug(self, original, binary, hpp, line_regions):
        """Displays the original image, the binary image, and the HPP histogram side-by-side."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title("Binarized")
        
        # Plot HPP sideways to align with the image rows
        y_coords = np.arange(len(hpp))
        axes[2].plot(hpp, y_coords, color='blue')
        axes[2].invert_yaxis() # Match image coordinate system (0 at top)
        axes[2].set_title("Horizontal Projection Profile")
        axes[2].set_xlabel("Number of Ink Pixels")
        
        # Draw red lines showing the calculated cuts
        for start_y, end_y in line_regions:
            axes[2].axhline(y=start_y, color='green', linestyle='--', alpha=0.6)
            axes[2].axhline(y=end_y, color='red', linestyle='--', alpha=0.6)
            
            axes[0].axhline(y=start_y, color='green', linestyle='--', alpha=0.6)
            axes[0].axhline(y=end_y, color='red', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Initialize the segmenter
    # Set debug=True to see the beautiful histogram plots!
    segmenter = HPPSegmenter(output_dir="hpp_output_lines", debug=True)
    
    # Replace with the path to one of your Modi script images
    # segmenter.segment_lines("path/to/modi_document.jpg")
    print("Script ready. Provide an image path to segmenter.segment_lines()")
