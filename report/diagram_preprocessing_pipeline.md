# Preprocessing Pipeline Diagram

Use this structure to build the diagram in Lucidchart. The pipeline is a **left-to-right horizontal flowchart** with 5 processing boxes connected by arrows. Each box has the stage name on top and the key parameters/formula below.

---

## Layout: Horizontal Flow (Left → Right)

```
┌─────────────────┐      ┌─────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────────┐
│                 │      │                     │      │                          │      │                             │      │                             │
│   GRAYSCALE     │─────▶│   BILATERAL         │─────▶│   ADAPTIVE GAUSSIAN      │─────▶│   SPECKLE REMOVAL           │─────▶│   HPP VARIANCE              │
│   CONVERSION    │      │   FILTER            │      │   THRESHOLDING           │      │   (Connected Components)    │      │   DESKEWING                 │
│                 │      │                     │      │                          │      │                             │      │                             │
└─────────────────┘      └─────────────────────┘      └──────────────────────────┘      └─────────────────────────────┘      └─────────────────────────────┘
```

---

## Box Details (Copy into each Lucid shape)

### Box 1: Grayscale Conversion
- **Title**: Grayscale Conversion
- **Subtitle**: Stage 1
- **Parameters**:
  - Formula: I = 0.299R + 0.587G + 0.114B
  - Output: Single-channel [0, 255]
- **Input label on arrow**: Raw BGR Image (3-channel)
- **Output label on arrow**: Grayscale Image

### Box 2: Bilateral Filter
- **Title**: Bilateral Filtering
- **Subtitle**: Stage 2 — Edge-Preserving Denoising
- **Parameters**:
  - d = 5 (neighbourhood diameter)
  - σ_color = 100
  - σ_space = 100
- **Tag**: Based on Kausadikar et al. (2025)
- **Output label on arrow**: Denoised Image

### Box 3: Adaptive Gaussian Thresholding
- **Title**: Adaptive Gaussian Thresholding
- **Subtitle**: Stage 3 — Binarization
- **Parameters**:
  - Block Size B = 61 (odd)
  - Constant C = 21
  - Method: ADAPTIVE_THRESH_GAUSSIAN_C
  - Auto-invert if mean < 127
- **Tag**: Based on Kausadikar et al. (2025)
- **Output label on arrow**: Binary Image (black ink, white bg)

### Box 4: Speckle Removal ⭐
- **Title**: Speckle Removal
- **Subtitle**: Stage 4 — Connected-Component Filtering
- **Parameters**:
  - Min Area: A_min = 100 px
  - Min Height: h_min = 10 px
  - Min Width: w_min = 7 px
  - Logic: Keep if (Area ≥ 100) OR (h ≥ 10 AND w ≥ 7)
- **Tag**: ⭐ NOVEL — Replaces manual Photoshop cleanup
- **Output label on arrow**: Cleaned Binary Image

### Box 5: HPP Variance Deskewing ⭐
- **Title**: Projection Profile Variance Deskewing
- **Subtitle**: Stage 5 — Skew Correction
- **Parameters**:
  - Angle Range: ±10°
  - Step: 0.5°
  - Candidates: 41 angles
  - Optimization: θ* = argmax Var(HPP(θ))
  - Downscale if width > 1000px
- **Tag**: ⭐ NOVEL — Automated deskewing
- **Output label on arrow**: Deskewed Binary Image (FINAL)

---

## Styling Suggestions for Lucidchart

| Element | Color | Notes |
|---------|-------|-------|
| Boxes 1-3 (standard stages) | Light blue fill (#D6EAF8) | From Kausadikar et al. baseline |
| Boxes 4-5 (novel stages) | Light green fill (#D5F5E3) with ⭐ badge | Highlight novel contributions |
| Arrows | Dark grey (#2C3E50) | Solid, with output labels |
| Parameter text | Monospace font, smaller size | Inside each box, below title |
| Input/Output labels | Italic, on the connecting arrows | Describe data format at each stage |

---

## Optional: Add a "Raw Input" start node and "To Segmentation" end node

```
( Raw Scan )  ──▶  [Box 1] ──▶ [Box 2] ──▶ [Box 3] ──▶ [Box 4] ──▶ [Box 5]  ──▶  ( To Phase 2: Segmentation )
```

Use rounded rectangles for the start/end terminators and standard rectangles for the processing boxes.
