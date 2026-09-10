"""
End-to-end preprocessing for archaic Modi script document images.

Reference:
Historic Scripts to Modern Vision (Kausadikar et al., 2025)

Pipeline implemented from the paper's dataset preprocessing section:
1) Grayscale conversion
2) Noise removal (bilateral filter, kernel size ~3)
3) Adaptive Gaussian thresholding
4) Deskewing

This script accepts a raw image path and writes a preprocessed image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    bilateral_d: int = 5
    bilateral_sigma_color: int = 100
    bilateral_sigma_space: int = 100
    adaptive_block_size: int = 61
    adaptive_c: int = 21
    final_black_text: bool = True
    
    # Remove tiny connected components (paper dots/speckles) after binarization.
    min_component_area: int = 100
    min_component_height: int = 10
    min_component_width: int = 7

    # Keep only components that look like "ink" in the original grayscale.
    # Lower value => stricter (keeps only very dark strokes).
    ink_mean_max: int = 50
    min_foreground_pixels_for_deskew: int = 500


def _validate_odd(value: int, name: str) -> int:
    if value % 2 == 0 or value <= 1:
        raise ValueError(f"{name} must be an odd integer > 1, got {value}")
    return value


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def remove_noise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    return cv2.bilateralFilter(
        gray,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )


def adaptive_gaussian_threshold(denoised: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    block_size = _validate_odd(cfg.adaptive_block_size, "adaptive_block_size")
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        cfg.adaptive_c,
    )
    # Paper visuals usually show dark text on light background; enforce if requested.
    if cfg.final_black_text and np.mean(binary) < 127:
        binary = 255 - binary
    return binary


def remove_speckle_components(binary: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Remove tiny isolated components created by aged-paper artifacts.

    Strategy:
    - Invert so foreground (text + noise) is white.
    - Run connected-components.
    - Keep components likely to be real strokes:
      * area >= min_component_area, OR
      * component has at least minimal stroke-like height+width.
    """
    inv = 255 - binary
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    cleaned_inv = np.zeros_like(inv)

    for idx in range(1, n_labels):  # skip background
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]

        keep = (
            area >= cfg.min_component_area
            or (h >= cfg.min_component_height and w >= cfg.min_component_width)
        )
        if keep:
            cleaned_inv[labels == idx] = 255

    return 255 - cleaned_inv


def keep_only_ink_components(binary: np.ndarray, gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Remove foreground components that are not dark in the grayscale image.

    This targets paper margin artifacts / background texture that thresholding can
    convert into black shapes, but whose underlying grayscale intensities are not
    ink-dark.
    """
    inv = 255 - binary  # foreground = white
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    kept_inv = np.zeros_like(inv)

    for idx in range(1, n_labels):
        component_mask = (labels == idx)
        if not np.any(component_mask):
            continue
        mean_intensity = float(gray[component_mask].mean())
        if mean_intensity <= float(cfg.ink_mean_max):
            kept_inv[component_mask] = 255

    return 255 - kept_inv


def estimate_skew_hpp(binary: np.ndarray, max_angle: float = 45.0, step: float = 0.5) -> float:
    """
    Estimates deskew angle by maximizing Horizontal Projection Profile variance.
    """
    # Foreground must be white (255) for projection
    fg = 255 - binary
    
    # Scale down for speed if large
    h, w = fg.shape[:2]
    if w > 1000:
        scale = 1000.0 / w
        new_w, new_h = int(w * scale), int(h * scale)
        fg = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = fg.shape[:2]
        
    center = (w / 2.0, h / 2.0)
    best_angle = 0.0
    max_variance = -1.0
    
    for angle in np.arange(-max_angle, max_angle + step, step):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(fg, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hpp = np.sum(rotated, axis=1)
        var = np.var(hpp)
        if var > max_variance:
            max_variance = var
            best_angle = angle
            
    return float(best_angle)


def deskew(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return binary

    h, w = binary.shape[:2]
    center: Tuple[float, float] = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated


def preprocess_image_array(image_bgr: np.ndarray, cfg: PreprocessConfig | None = None, return_intermediates: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    cfg = cfg or PreprocessConfig()
    gray = to_grayscale(image_bgr)
    denoised = remove_noise(gray, cfg)
    binary = adaptive_gaussian_threshold(denoised, cfg)
    cleaned = remove_speckle_components(binary, cfg)
    ink_only = cleaned

    angle = estimate_skew_hpp(ink_only)
    deskewed = deskew(ink_only, angle)
    
    if return_intermediates:
        intermediates = {
            "grayscale": gray,
            "denoised": denoised,
            "binary": binary,
            "cleaned": cleaned,
            "deskewed": deskewed,
        }
        return deskewed, intermediates

    return deskewed


def preprocess_image_file(
    input_path: str,
    output_path: str | None = None,
    cfg: PreprocessConfig | None = None,
) -> str:
    cfg = cfg or PreprocessConfig()
    image = load_image(input_path)
    processed = preprocess_image_array(image, cfg)

    in_path = Path(input_path)
    if output_path is None:
        output_path = str(in_path.with_name(f"{in_path.stem}_preprocessed.png"))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), processed)
    if not ok:
        raise IOError(f"Failed to write preprocessed image to: {out_path}")
    return str(out_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess archaic Modi script image."
    )
    parser.add_argument("--input", required=True, help="Path to raw input image")
    parser.add_argument("--output", default=None, help="Path to save preprocessed image")
    parser.add_argument(
        "--block-size",
        type=int,
        default=35,
        help="Adaptive threshold block size (odd integer > 1)",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=11,
        help="Adaptive threshold constant C",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=18,
        help="Minimum connected-component area to keep",
    )
    parser.add_argument(
        "--ink-mean-max",
        type=int,
        default=170,
        help="Max mean grayscale intensity for a foreground component to be kept (lower=stricter)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = PreprocessConfig(
        adaptive_block_size=args.block_size,
        adaptive_c=args.c,
        min_component_area=args.min_component_area,
        ink_mean_max=args.ink_mean_max,
    )
    out = preprocess_image_file(args.input, args.output, cfg)
    print(f"Saved preprocessed image: {out}")


if __name__ == "__main__":
    main()
