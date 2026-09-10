# Software Architecture Diagram

This diagram shows the interaction between Python files, classes, functions, and data flow across the pipeline. Build it in Lucidchart as a **top-down layered diagram** with 4 horizontal layers.

---

## Layout: 4 Layers (Top → Bottom)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        LAYER 1: GUI CONTROLLER                              ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────┐        ║
║   │  phase1and2.py                                                  │        ║
║   │  Class: EndToEndApp                                             │        ║
║   │                                                                 │        ║
║   │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐     │        ║
║   │  │ open_image() │  │ run_pipeline()│  │ on_slider_change()│     │        ║
║   │  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘     │        ║
║   │         │                 │                    │                │        ║
║   └─────────│─────────────────│────────────────────│────────────────┘        ║
╚═════════════│═════════════════│════════════════════│═════════════════════════╝
              │                 │                    │
              ▼                 ▼                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LAYER 2: PREPROCESSING ENGINE                           ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────┐        ║
║   │  preprocessing/preprocess.py                                    │        ║
║   │                                                                 │        ║
║   │  Config: PreprocessConfig (dataclass)                           │        ║
║   │                                                                 │        ║
║   │  Orchestrator: preprocess_image_array(image_bgr, cfg)           │        ║
║   │       │                                                         │        ║
║   │       ├──▶ to_grayscale()                                       │        ║
║   │       ├──▶ remove_noise()               ── cv2.bilateralFilter  │        ║
║   │       ├──▶ adaptive_gaussian_threshold() ── cv2.adaptiveThresh  │        ║
║   │       ├──▶ remove_speckle_components()  ── cv2.connectedComps   │        ║
║   │       └──▶ estimate_skew_hpp() + deskew() ── cv2.warpAffine    │        ║
║   │                                                                 │        ║
║   │  Returns: (deskewed_image, intermediates_dict)                  │        ║
║   └──────────────────────────────────┬──────────────────────────────┘        ║
╚══════════════════════════════════════│═══════════════════════════════════════╝
                                       │
                          deskewed image passed down
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
╔═════════════════════════════════╗  ╔═════════════════════════════════════════╗
║   LAYER 3a: FREQUENCY TUNER    ║  ║   LAYER 3b: SEGMENTATION ENGINE         ║
║                                 ║  ║                                         ║
║  fft_analyzer.py                ║  ║  segmentation_engine.py                 ║
║                                 ║  ║                                         ║
║  Class: FFTAnalyzer (static)    ║  ║  Base: SegmentationEngine               ║
║                                 ║  ║    └─ preprocess(image_path)            ║
║  Method:                        ║  ║                                         ║
║  calculate_optimal_parameters() ║  ║  Class: AnisotropicGaussianSegmenter    ║
║       │                         ║  ║    └─ segment(image_path,               ║
║       ├─ HPP computation        ║  ║          x_stretch, y_stretch,          ║
║       ├─ np.correlate (ACF)     ║  ║          thresh_val, pad)               ║
║       ├─ find_peaks (SciPy)     ║  ║       │                                ║
║       └─ Parameter mapping      ║  ║       ├─ cv2.GaussianBlur (aniso)      ║
║                                 ║  ║       ├─ cv2.threshold                 ║
║  Returns: dict{                 ║  ║       ├─ cv2.findContours              ║
║    spatial_period_px,           ║  ║       └─ Bounding box filter + sort     ║
║    x_stretch,                   ║  ║                                         ║
║    y_stretch,                   ║  ║  Returns: (original, blob_mask,         ║
║    savgol_window,               ║  ║            line_regions, pad)           ║
║    ink_threshold                ║  ║                                         ║
║  }                              ║  ║                                         ║
╚════════════════╤════════════════╝  ╚══════════════════╤══════════════════════╝
                 │                                      │
                 │  σ_x, σ_y                            │  line_regions
                 └──────────┬───────────────────────────┘
                            │
                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        LAYER 4: OUTPUT                                      ║
║                                                                              ║
║   ┌──────────────────────┐    ┌──────────────────────────────────────┐       ║
║   │ Preprocessing Grid   │    │ Extracted Lines (scrollable)         │       ║
║   │ 2×3 image slots      │    │ Individual line crops from regions   │       ║
║   └──────────────────────┘    └──────────────────────────────────────┘       ║
║                                                                              ║
║   Rendered via Pillow (PIL.ImageTk.PhotoImage) into Tkinter Labels          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Arrows (Label each in Lucid)

| From | To | Arrow Label | Type |
|------|----|-------------|------|
| EndToEndApp.open_image() | preprocess_image_array() | Raw BGR image (np.ndarray) | Solid |
| preprocess_image_array() | EndToEndApp | intermediates dict + deskewed image | Solid |
| EndToEndApp.run_pipeline() | FFTAnalyzer | Temp image path (saves deskewed to disk) | Dashed |
| FFTAnalyzer | EndToEndApp | {x_stretch, y_stretch, ...} | Dashed |
| EndToEndApp | AnisotropicGaussianSegmenter.segment() | image_path + σ_x, σ_y, pad | Solid |
| AnisotropicGaussianSegmenter | EndToEndApp | line_regions list | Solid |
| EndToEndApp | Tkinter Labels | PIL.ImageTk.PhotoImage | Solid (thin) |

---

## Iterative Loop Annotation

Draw a **dashed loop arrow** from `AnisotropicGaussianSegmenter` output back to `FFTAnalyzer` input with the label:

> "Iterative Masking: line regions → mask image → re-run ACF (×3)"

This represents the self-correcting loop described in Section 3.3 of the report.

---

## Threading Annotation

Add a small badge or callout near `run_pipeline()` and `on_slider_change()`:

> "Runs in background threading.Thread — GUI updates via root.after()"

---

## Styling Suggestions

| Layer | Color | Notes |
|-------|-------|-------|
| Layer 1 (GUI) | Light grey (#F2F3F4) | Controller layer |
| Layer 2 (Preprocessing) | Light blue (#D6EAF8) | 5 internal function boxes |
| Layer 3a (FFT Analyzer) | Orange (#FAD7A0) | Frequency domain |
| Layer 3b (Segmenter) | Green (#D5F5E3) | Spatial domain |
| Layer 4 (Output) | White | Final display |
| All arrows | Dark grey (#2C3E50) | Solid for data, dashed for control |
| Class names | Bold monospace | Inside each module box |
| Function names | Regular monospace, smaller | Indented under class |
