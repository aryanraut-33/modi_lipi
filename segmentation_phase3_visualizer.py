#!/usr/bin/env python3
"""
Modi ShatVidhi: Phase 3 Character Segmentation Verification & Visualizer
Processes input raw Modi manuscripts through Phase 1 (Preprocessing), Phase 2 (Line Localization),
and Phase 3 (Glyph Segmentation), outputting quantitative evaluation stats and sample visual cuts.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Path injections
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "preprocessing"))
sys.path.append(os.path.join(ROOT, "segmentation", "line_level", "implementation", "Image Processing", "Suite"))
sys.path.append(os.path.join(ROOT, "segmentation", "character_level"))

from preprocess import preprocess_image_array, PreprocessConfig
from segmentation_engine import AnisotropicGaussianSegmenter
from fft_analyzer import FFTAnalyzer
from character_segmenter import CharacterSegmenter, CharSegConfig

def run_phase3_evaluation(image_path: str, output_dir: str = "output_phase3_debug"):
    os.makedirs(output_dir, exist_ok=True)
    img_name = Path(image_path).stem
    print(f"\n=======================================================")
    print(f" Processing: {Path(image_path).name}")
    print(f"=======================================================")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load {image_path}")
        return

    # 1. Phase 1 Preprocessing
    print("[Phase 1] Running Document Enhancement & Deskewing...")
    cfg = PreprocessConfig()
    clean_gray, _ = preprocess_image_array(img_bgr, cfg, return_intermediates=True)
    temp_preprocessed = os.path.join(output_dir, f"{img_name}_preprocessed.png")
    cv2.imwrite(temp_preprocessed, clean_gray)

    # 2. Phase 2 Line Segmentation
    print("[Phase 2] Running Text-Line Localization (FFT Auto-Tuning)...")
    fft_results = FFTAnalyzer.calculate_optimal_parameters(temp_preprocessed)
    x_stretch = fft_results.get("x_stretch", 41) if fft_results else 41
    y_stretch = fft_results.get("y_stretch", 5) if fft_results else 5
    
    segmenter = AnisotropicGaussianSegmenter()
    _, _, line_regions, _ = segmenter.segment(temp_preprocessed, x_stretch=x_stretch, y_stretch=y_stretch)
    print(f"  --> Extracted {len(line_regions)} text lines.")

    # 3. Phase 3 Character Segmentation
    print("[Phase 3] Running Glyph-Level Character Segmentation (Shirorekha/Baseline + Cluster Disambiguation)...")
    char_cfg = CharSegConfig(target_glyph_size=(64, 64))
    char_segmenter = CharacterSegmenter(config=char_cfg)

    all_glyphs = char_segmenter.segment_document(clean_gray, line_regions)

    total_glyphs = sum(len(g_list) for g_list in all_glyphs)
    print(f"\n--- Segmentation Summary for {img_name} ---")
    print(f"  Total Lines Segmented: {len(line_regions)}")
    print(f"  Total Characters/Glyphs Segmented: {total_glyphs}")

    # 4. Save Visual Artifacts
    for line_idx, (sy, ey) in enumerate(line_regions):
        sy_clamped, ey_clamped = max(0, sy), min(clean_gray.shape[0], ey)
        line_crop = clean_gray[sy_clamped:ey_clamped, :]
        glyphs = all_glyphs[line_idx]

        # Draw annotated line
        annotated = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2BGR)
        _, debug_info = char_segmenter.segment_line(line_crop, line_id=line_idx)
        zinfo = debug_info.get("zonal_info", {})
        ls = zinfo.get("shirorekha_y", 0)
        lb = zinfo.get("baseline_y", line_crop.shape[0])

        # Blue = Shirorekha, Orange = Baseline
        cv2.line(annotated, (0, ls), (line_crop.shape[1], ls), (255, 120, 0), 1)
        cv2.line(annotated, (0, lb), (line_crop.shape[1], lb), (0, 140, 255), 1)

        # Green = Bounding Boxes
        for g in glyphs:
            x1, y1, x2, y2 = g.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 1)

        line_out_path = os.path.join(output_dir, f"{img_name}_line_{line_idx+1}_segmented.png")
        cv2.imwrite(line_out_path, annotated)

        # Save first 5 normalized 64x64 tensors as sample
        if line_idx == 0 and glyphs:
            grid_samples = []
            for g in glyphs[:min(10, len(glyphs))]:
                norm_tensor = g.to_fixed_size((64, 64))
                grid_samples.append(norm_tensor)
            if grid_samples:
                grid_img = np.hstack(grid_samples)
                cv2.imwrite(os.path.join(output_dir, f"{img_name}_line1_glyphs_64x64.png"), grid_img)

    print(f"  Artifacts saved to: {output_dir}/")

def main():
    images = sorted(list(Path("images_raw").glob("*.jpg")) + list(Path("images_raw").glob("*.png")))
    if not images:
        print("No raw images found in images_raw/")
        return

    print(f"Found {len(images)} test manuscripts in images_raw/.")
    for img_p in images:
        run_phase3_evaluation(str(img_p))

if __name__ == "__main__":
    main()
