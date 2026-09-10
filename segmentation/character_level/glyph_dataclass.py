import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

@dataclass
class SegmentedGlyph:
    """
    Represents an extracted individual glyph/character unit from a Modi text line.
    Preserves multi-zonal information (Upper Matra, Base Consonant, Lower Modifier)
    and provides normalized tensor outputs for downstream Phase 4 recognition.
    """
    glyph_id: int
    line_id: int
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in text-line coordinates
    full_glyph_img: np.ndarray       # Complete character image (Upper matras + Headline + Body + Descenders)
    base_glyph_img: Optional[np.ndarray] = None  # Middle zone crop
    upper_zone_img: Optional[np.ndarray] = None  # Ascenders / Top Matras
    lower_zone_img: Optional[np.ndarray] = None  # Descenders / Bottom Matras
    has_upper_modifier: bool = False
    has_lower_modifier: bool = False
    has_attached_kana: bool = False
    aspect_ratio: float = 1.0        # width / height
    density: float = 0.0             # Foreground pixel ratio
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_fixed_size(self, target_size: Tuple[int, int] = (64, 64), pad_value: int = 255) -> np.ndarray:
        """
        Normalizes the complete character glyph into a fixed square dimension (default 64x64)
        using aspect-ratio-preserving padding to prevent stroke distortion for Phase 4 CNNs.
        Ensures all upper matras, headlines, consonants, and descenders are preserved intact.

        Args:
            target_size: Desired (width, height), typically (64, 64).
            pad_value: Background padding pixel value (255 for white background).

        Returns:
            Normalized np.ndarray of shape (target_size[1], target_size[0]).
        """
        # Always use the complete character image (full_glyph_img) so matras/ascenders are never cut
        img = self.full_glyph_img
        if img is None or img.size == 0:
            img = self.base_glyph_img

        if img is None or img.size == 0:
            return np.full(target_size, pad_value, dtype=np.uint8)

        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Leave a 4-pixel border margin inside the 64x64 square
        usable_w = max(1, target_w - 6)
        usable_h = max(1, target_h - 6)

        # Scaling factor to fit within canvas while maintaining true aspect ratio
        scale = min(usable_w / max(w, 1), usable_h / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Create padded canvas and center the glyph
        padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return padded
