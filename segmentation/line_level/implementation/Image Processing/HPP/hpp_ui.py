import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from hpp_segmenter import HPPSegmenter

st.set_page_config(page_title="HPP Segmentation UI", layout="wide")

st.title("Historical Document Line Segmentation (HPP)")
st.write("Tune the parameters on the left to perfectly segment lines using Horizontal Projection Profile.")

# --- Sidebar Parameters ---
st.sidebar.header("1. Preprocessing Parameters")
blur_ksize = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2, help="Removes high-frequency noise")
block_size = st.sidebar.slider("Adaptive Thresh Block Size", 3, 51, 15, step=2, help="Neighborhood size for thresholding")
c_val = st.sidebar.slider("Adaptive Thresh C Value", 0, 30, 10, help="Constant subtracted from the mean")

st.sidebar.header("2. HPP Parameters")
savgol_window = st.sidebar.slider("Savgol Filter Window", 5, 51, 21, step=2, help="Smoothing window for histogram. Larger = smoother peaks")
ink_threshold = st.sidebar.slider("Ink Threshold", 1, 50, 5, help="Minimum ink pixels required to consider a row as text")
min_line_height = st.sidebar.slider("Min Line Height (px)", 5, 50, 15, help="Ignore line segments smaller than this")
pad = st.sidebar.slider("Crop Padding (px)", 0, 20, 5, help="Padding added to the top and bottom of each segmented line")

st.sidebar.markdown("---")
st.sidebar.info("Upload an image on the right to see the results.")

# --- Main App ---
uploaded_file = st.file_uploader("Upload Document Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Save temporarily to pass to the segmenter
    temp_path = "temp_upload.jpg"
    cv2.imwrite(temp_path, img)
    
    # Initialize Segmenter with UI parameters
    segmenter = HPPSegmenter(
        output_dir="temp_output", 
        debug=False, # We'll do the visualization manually in streamlit
        blur_ksize=blur_ksize,
        block_size=block_size,
        c_val=c_val,
        savgol_window=savgol_window,
        ink_threshold=ink_threshold,
        min_line_height=min_line_height,
        pad=pad
    )
    
    st.subheader("Results")
    
    with st.spinner("Processing image..."):
        # Run preprocessing
        original, binary = segmenter.preprocess(temp_path)
        
        # Run HPP
        hpp = segmenter.compute_hpp(binary)
        
        # Smooth HPP
        from scipy.signal import savgol_filter
        smoothed_hpp = savgol_filter(hpp, window_length=savgol_window, polyorder=3)
        smoothed_hpp[smoothed_hpp < 0] = 0
        
        # Detect Lines
        is_text_row = smoothed_hpp > ink_threshold
        line_regions = []
        in_line = False
        start_y = 0
        
        for y, is_text in enumerate(is_text_row):
            if is_text and not in_line:
                start_y = y
                in_line = True
            elif not is_text and in_line:
                end_y = y
                in_line = False
                if (end_y - start_y) > min_line_height:
                    line_regions.append((start_y, end_y))
                    
        if in_line and (len(is_text_row) - start_y) > min_line_height:
            line_regions.append((start_y, len(is_text_row)))
            
        # --- Visualization ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Document**")
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown("**Binarized (Ink Mask)**")
            st.image(binary, use_container_width=True)
            
        with col3:
            st.markdown(f"**HPP Histogram ({len(line_regions)} lines found)**")
            fig, ax = plt.subplots(figsize=(4, original.shape[0]/100))
            y_coords = np.arange(len(smoothed_hpp))
            ax.plot(smoothed_hpp, y_coords, color='blue')
            ax.invert_yaxis()
            ax.set_xlabel("Ink Pixels")
            
            # Draw cutlines
            for sy, ey in line_regions:
                ax.axhline(y=sy, color='green', linestyle='--', alpha=0.6)
                ax.axhline(y=ey, color='red', linestyle='--', alpha=0.6)
            
            ax.margins(y=0)
            st.pyplot(fig)
            
        # Display cropped lines
        st.markdown("---")
        st.subheader(f"Extracted Lines ({len(line_regions)})")
        
        # Show first 10 lines to prevent crashing the browser on huge documents
        max_lines_to_show = 10
        for idx, (start_y, end_y) in enumerate(line_regions[:max_lines_to_show]):
            y1 = max(0, start_y - pad)
            y2 = min(original.shape[0], end_y + pad)
            line_crop = original[y1:y2, :]
            
            st.image(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB), caption=f"Line {idx+1}")
            
        if len(line_regions) > max_lines_to_show:
            st.info(f"...and {len(line_regions) - max_lines_to_show} more lines not shown for performance.")
            
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
