import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from segmentation_engine import (
    HPPSegmenter, 
    GapAnalysisSegmenter, 
    AnisotropicGaussianSegmenter, 
    EdgeConnectivitySegmenter, 
    SeamCarvingSegmenter
)
from auto_tuner import AutoTuner

st.set_page_config(page_title="Unified Segmentation Suite", layout="wide")

st.title("Unified Unsupervised Line Segmentation Suite")
st.write("Compare 5 state-of-the-art classical computer vision techniques for historical document line extraction.")

# --- Session State Initialization for Auto-Tuner ---
if "best_params" not in st.session_state:
    st.session_state.best_params = {}

# --- Sidebar ---
st.sidebar.header("1. Select Algorithm")
algorithm = st.sidebar.selectbox(
    "Segmentation Method",
    (
        "Anisotropic Gaussian Blobs (Barakat 2020)",
        "Horizontal Projection Profile (Baseline)",
        "Statistical Gap Analysis (Mukherjee 2022)"
    )
)

st.sidebar.header("2. Preprocessing")
blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, st.session_state.best_params.get("blur_ksize", 5), step=2)
block_size = st.sidebar.slider("Adaptive Thresh Block", 3, 51, st.session_state.best_params.get("block_size", 15), step=2)
c_val = st.sidebar.slider("Adaptive Thresh C Value", 0, 30, st.session_state.best_params.get("c_val", 10))

st.sidebar.header("3. Algorithm-Specific Parameters")

kwargs = {
    "blur_ksize": blur_ksize,
    "block_size": block_size,
    "c_val": c_val
}

# --- Dynamic Optimization UI Integration ---
from fft_analyzer import FFTAnalyzer
from vlm_agent import VLMAgent

st.sidebar.markdown("---")
run_auto_tune = False
run_fft = False
run_vlm = False

if algorithm in ["Horizontal Projection Profile (Baseline)", "Anisotropic Gaussian Blobs (Barakat 2020)"]:
    st.sidebar.subheader("Dynamic Parameter Tuning")
    
    st.sidebar.markdown("**Option 1: VLM Agent (OpenRouter)**")
    run_vlm = st.sidebar.button("🧠 Run Vision LLM Analysis")
    
    st.sidebar.markdown("**Option 2: Auto-Tuning (Heuristic Loop)**")
    run_auto_tune = st.sidebar.button("🤖 Run Heuristic Auto-Tuner")
    
    st.sidebar.markdown("**Option 3: Frequency Domain (FFT)**")
    run_fft = st.sidebar.button("⚡ Run FFT Analyzer")

st.sidebar.markdown("---")

if algorithm == "Horizontal Projection Profile (Baseline)":
    segmenter_class = HPPSegmenter
    kwargs["savgol_window"] = st.sidebar.slider("Savgol Window", 5, 51, st.session_state.best_params.get("savgol_window", 21), step=2)
    kwargs["ink_threshold"] = st.sidebar.slider("Ink Threshold", 1, 50, st.session_state.best_params.get("ink_threshold", 5))
    kwargs["min_line_height"] = st.sidebar.slider("Min Line Height", 5, 50, st.session_state.best_params.get("min_line_height", 15))
    kwargs["pad"] = st.sidebar.slider("Padding", 0, 20, st.session_state.best_params.get("pad", 5))
    
    auto_tune_grid = {
        "blur_ksize":      [3, 5, 7],
        "block_size":      [11, 15, 21, 31],
        "c_val":           [3, 7, 10, 15],
        "savgol_window":   [11, 21, 31, 41, 51],
        "ink_threshold":   [2, 5, 8, 12, 18],
        "min_line_height": [10, 15, 20],
        "pad":             [5]
    }
    
elif algorithm == "Anisotropic Gaussian Blobs (Barakat 2020)":
    segmenter_class = AnisotropicGaussianSegmenter
    kwargs["x_stretch"] = st.sidebar.slider("X-Axis Blur (Horizontal Smear)", 1, 101, st.session_state.best_params.get("x_stretch", 41), step=2)
    kwargs["y_stretch"] = st.sidebar.slider("Y-Axis Blur (Vertical Smear)", 1, 21, st.session_state.best_params.get("y_stretch", 5), step=2)
    kwargs["thresh_val"] = st.sidebar.slider("Blob Threshold", 10, 200, st.session_state.best_params.get("thresh_val", 50))
    kwargs["pad"] = st.sidebar.slider("Padding", 0, 20, st.session_state.best_params.get("pad", 5))
    
    auto_tune_grid = {
        "blur_ksize":  [3, 5, 7],
        "block_size":  [11, 15, 21, 31],
        "c_val":       [5, 10, 15],
        "x_stretch":   [21, 31, 41, 51, 61, 71, 81, 101],
        "y_stretch":   [3, 5, 7, 9, 11, 13],
        "thresh_val":  [20, 30, 40, 50, 60, 80],
        "pad":         [5]
    }
    
elif algorithm == "Statistical Gap Analysis (Mukherjee 2022)":
    segmenter_class = GapAnalysisSegmenter
    st.sidebar.info("This algorithm uses KMeans clustering to automatically learn the gap threshold.")
    kwargs["min_line_height"] = st.sidebar.slider("Min Line Height", 5, 50, 15)
    kwargs["pad"] = st.sidebar.slider("Padding", 0, 20, 5)

# --- Main App ---
uploaded_file = st.file_uploader("Upload Document Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    temp_path = "temp_suite_upload.jpg"
    cv2.imwrite(temp_path, img)
    
    st.subheader(f"Results: {algorithm}")
    
    # Check if Optimization was clicked
    if run_auto_tune:
        with st.spinner("🤖 Auto-Tuning! Searching parameter combinations mathematically..."):
            tuner = AutoTuner(segmenter_class)
            best_p, best_score, best_reg = tuner.tune(temp_path, auto_tune_grid)
            
            if best_p is not None:
                st.session_state.best_params.update(best_p)
                st.success(f"Heuristic Auto-Tuner optimized parameters (Score: {best_score:.2f})!")
                st.rerun()
            else:
                st.error("Auto-Tuner failed to find a valid segmentation.")
                
    elif run_fft:
        with st.spinner("⚡ Running ACF Frequency Analysis..."):
            fft_results = FFTAnalyzer.calculate_optimal_parameters(temp_path)
            if fft_results:
                # Apply ALL computed parameters at once
                updatable_keys = ["y_stretch", "x_stretch", "savgol_window", "ink_threshold"]
                for key in updatable_keys:
                    if key in fft_results:
                        st.session_state.best_params[key] = fft_results[key]
                period = fft_results['spatial_period_px']
                st.success(f"✅ ACF detected spatial period: **{period:.1f}px**  →  y_stretch={fft_results['y_stretch']}, x_stretch={fft_results['x_stretch']}, savgol={fft_results['savgol_window']}, ink_thresh={fft_results['ink_threshold']}")
                st.rerun()
            else:
                st.error("ACF Analysis failed — could not find a dominant line period.")
                
    elif run_vlm:
        with st.spinner("🧠 Sending image to Vision-Language Model..."):
            vlm_results = VLMAgent.query_vlm_for_parameters(temp_path)
            if vlm_results:
                st.session_state.best_params.update(vlm_results)
                st.success(f"VLM Agent returned parameters! Sliders updated.")
                st.rerun()
            else:
                st.error("VLM Agent failed to analyze the image.")
    
    with st.spinner(f"Running {algorithm}..."):
        segmenter = segmenter_class()
        original, debug_img, line_regions, pad = segmenter.segment(temp_path, **kwargs)
        
        # --- Visualization ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Document**")
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown("**Algorithm Intermediate / Debug View**")
            if len(debug_img.shape) == 2:
                st.image(debug_img, use_container_width=True)
            else:
                st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        st.markdown("---")
        st.subheader(f"Extracted Lines ({len(line_regions)})")
        
        max_lines_to_show = 10
        for idx, (start_y, end_y) in enumerate(line_regions[:max_lines_to_show]):
            y1 = max(0, start_y - pad)
            y2 = min(original.shape[0], end_y + pad)
            line_crop = original[y1:y2, :]
            st.image(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB), caption=f"Line {idx+1}")
            
        if len(line_regions) > max_lines_to_show:
            st.info(f"...and {len(line_regions) - max_lines_to_show} more lines not shown for performance.")
            
    if os.path.exists(temp_path):
        os.remove(temp_path)
