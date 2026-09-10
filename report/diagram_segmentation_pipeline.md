# Segmentation Pipeline Diagram

Use this structure to build the diagram in Lucidchart. The diagram has **two main sections**: the outer **Iterative Masking Loop** (a cycle) and the inner **Anisotropic Gaussian Blob Engine** (a linear flow).

---

## Overall Layout

The diagram should be read as a **cycle with a linear core**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ITERATIVE MASKING LOOP (K = 3 iterations)                        │
│                                                                                                │
│   ┌───────────────┐      ┌──────────────────────┐      ┌──────────────────────────────────┐   │
│   │               │      │                      │      │    ANISOTROPIC GAUSSIAN BLOB     │   │
│   │  ACF / FFT    │─────▶│  PARAMETER           │─────▶│    SEGMENTATION ENGINE            │   │
│   │  ANALYZER     │      │  DERIVATION           │      │                                  │   │
│   │               │      │                      │      │  [See detail below]              │   │
│   └───────────────┘      └──────────────────────┘      └──────────────┬───────────────────┘   │
│          ▲                                                            │                       │
│          │                                                            ▼                       │
│          │                                              ┌──────────────────────────┐          │
│          │                                              │                          │          │
│          └──────────────────────────────────────────────│   MASK CONSTRUCTION      │          │
│                          Cleaned image fed back         │   (Erase non-text areas) │          │
│                                                         └──────────────────────────┘          │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │  EXTRACTED LINES   │
                                    │  (y_start, y_end)  │
                                    └────────────────────┘
```

---

## Box Details

### Box A: ACF / FFT Analyzer
- **Title**: Autocorrelation Frequency Analyzer
- **Parameters**:
  - Input: Binary image (or masked image after iteration 1)
  - Binarize: block=15, C=10
  - HPP: h(y) = Σ I_bin(x,y) / 255
  - ACF: R(τ) = Σ h̃(y) · h̃(y+τ)
  - Peak detection: prominence = 5% of max, distance = 15px
  - Output: Spatial Period P (pixels)

### Box B: Parameter Derivation
- **Title**: Parameter Mapping
- **Parameters**:
  - σ_y = max(3, ⌊P/6⌋)  — forced odd
  - σ_x = max(21, ⌊1.5·P⌋) — forced odd
  - Both values passed to segmenter

### Box C: Anisotropic Gaussian Blob Engine (expand into sub-flow)
- **Title**: Anisotropic Gaussian Blob Segmenter
- **Sub-steps** (show as a mini horizontal flow inside this box):

```
  [Binary Mask]  ──▶  [Directional Gaussian Blur]  ──▶  [Threshold at T=50]  ──▶  [Find Contours]  ──▶  [Filter + Sort]
   (Inverted)          (σ_x, σ_y)                       Binary blobs             Bounding boxes         Line regions
```

- **Filter criteria**: width > 10% of image width AND height > 10px
- **Output**: List of (y_start, y_end) tuples, sorted top-to-bottom

### Box D: Mask Construction
- **Title**: Noise Masking for Next Iteration
- **Logic**:
  - Create blank white image (same size)
  - For each detected line region (y_s, y_e):
    - Copy original pixels from y_s to y_e
  - Everything outside line regions stays white
  - This cleaned image is fed BACK to the ACF Analyzer
- **Tag**: Removes margin noise, page borders, inter-paragraph gaps

### Output Node: Extracted Lines
- **Title**: Final Extracted Lines
- **Content**: Individual line crops ready for Phase 3 (Recognition)

---

## Loop Annotations

Add these labels on the loop arrows:

| Arrow | Label |
|-------|-------|
| ACF → Parameter Derivation | "Spatial Period P" |
| Parameter Derivation → Segmenter | "σ_x, σ_y" |
| Segmenter → Mask Construction | "Line regions {(y_s, y_e)}" |
| Mask Construction → ACF | "Cleaned masked image (iteration i+1)" |
| Loop border annotation | "Repeat K times (default K=3)" |

---

## Styling Suggestions for Lucidchart

| Element | Color | Notes |
|---------|-------|-------|
| Outer loop border | Dashed dark blue (#2C3E50) | Large rounded rectangle enclosing all 4 boxes |
| ACF Analyzer box | Orange fill (#FAD7A0) | Frequency analysis |
| Parameter Derivation box | Yellow fill (#FEF9E7) | Mathematical mapping |
| Segmenter Engine box | Blue fill (#D6EAF8) | Core algorithm — largest box |
| Mask Construction box | Light purple (#E8DAEF) | Feedback mechanism |
| Output node | Green fill (#D5F5E3) | Final output |
| Loop arrows | Dashed dark blue | Distinguish from linear arrows |
| Linear arrows | Solid dark grey | Standard flow |
| Iteration counter | Badge in corner of loop border | "K = 3" |

---

## Input/Output Context

Add small labels showing where this diagram connects to the broader pipeline:

```
FROM Phase 1:                                                    TO Phase 3:
Deskewed Binary Image  ──▶  [ SEGMENTATION PIPELINE ]  ──▶  Individual Line Crops
                                                              (for VLM / OCR)
```
