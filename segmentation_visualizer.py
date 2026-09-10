import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys

# Add paths to allow importing your existing FFT Analyzer and Segmenter logic
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "segmentation", "line_level", "implementation", "Image Processing", "Suite"))

class SegmentationVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stage 2: Iterative Line Segmentation Visualizer")
        self.root.geometry("1400x900")
        
        self.input_path = None
        self.original_img = None
        self.current_images = {}
        
        self._build_ui()
        
    def _build_ui(self):
        # ---------------- Control Panel ----------------
        control_panel = ttk.Frame(self.root, padding=10)
        control_panel.pack(fill=tk.X)
        
        ttk.Button(control_panel, text="Open Deskewed Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_panel, text="Run Segmentation", command=self.run_segmentation).pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Load a preprocessed/deskewed image to start.")
        ttk.Label(control_panel, textvariable=self.status_var, font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=20)
        
        # ---------------- Parameter Tuning ----------------
        tuning_frame = ttk.LabelFrame(control_panel, text="Segmentation Parameters (Param Map)")
        tuning_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(tuning_frame, text="σ_x (X-Stretch):").grid(row=0, column=0, padx=5)
        self.x_var = tk.IntVar(value=41)
        tk.Scale(tuning_frame, variable=self.x_var, from_=5, to=301, orient=tk.HORIZONTAL).grid(row=0, column=1, padx=5)
        
        ttk.Label(tuning_frame, text="σ_y (Y-Stretch):").grid(row=0, column=2, padx=5)
        self.y_var = tk.IntVar(value=5)
        tk.Scale(tuning_frame, variable=self.y_var, from_=1, to=51, orient=tk.HORIZONTAL).grid(row=0, column=3, padx=5)
        
        self.use_fft = tk.BooleanVar(value=True)
        ttk.Checkbutton(tuning_frame, text="Use ACF Analyzer (FFT Auto-Tune)", variable=self.use_fft).grid(row=0, column=4, padx=15)

        # ---------------- Pipeline Visualization Grid ----------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="Segmentation Architecture Steps")
        
        # Configure a 3x2 Grid for the 6 stages
        for i in range(3):
            self.pipeline_tab.columnconfigure(i, weight=1)
            self.pipeline_tab.rowconfigure(1, weight=1)
            self.pipeline_tab.rowconfigure(3, weight=1)
            
        labels = [
            "1. Deskewed Image", 
            "2. Gaussian Blur (Anisotropic Smear)", 
            "3. Thresholding (Solid Blobs)",
            "4. Contours (Bounding Regions)",
            "5. Mask Build (Iterative Base)",
            "6. Extracted Lines (Final Preview)"
        ]
        
        self.image_labels = []
        for i, text in enumerate(labels):
            row = (i // 3) * 2
            col = i % 3
            ttk.Label(self.pipeline_tab, text=text, font=("Arial", 12, "bold")).grid(row=row, column=col, pady=10)
            lbl = ttk.Label(self.pipeline_tab, cursor="hand2")
            lbl.grid(row=row+1, column=col, sticky="nsew", padx=10, pady=5)
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_image_click(idx))
            self.image_labels.append(lbl)

    def _to_photo(self, img, max_w=400, max_h=300):
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Ensure we always pass RGB to Pillow
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)
        
    def open_image(self):
        path = filedialog.askopenfilename(title="Select Deskewed Preprocessed Image")
        if not path:
            return
            
        self.input_path = path
        self.original_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if self.original_img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
            
        self._update_display(0, self.original_img)
        
        # Clear previous pipeline run visuals
        for i in range(1, 6):
            self.image_labels[i].configure(image="")
            self.image_labels[i].image = None
            
        self.status_var.set(f"Loaded: {os.path.basename(path)}. Ready to run segmentation.")
        
    def _update_display(self, index, img):
        self.current_images[index] = img
        photo = self._to_photo(img)
        if photo:
            self.image_labels[index].configure(image=photo)
            self.image_labels[index].image = photo

    def _on_image_click(self, index):
        if index not in self.current_images or self.current_images[index] is None:
            return
            
        img = self.current_images[index]
        self._show_enlarged(img, f"Stage {index + 1} (Full Resolution)")
        
    def _show_enlarged(self, img, title="Enlarged Image"):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("1000x800")
        
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="💾 Download / Save Full Image", command=lambda: self._save_image(img)).pack(side=tk.RIGHT, padx=10)
        
        # Scrollable canvas for large image
        canvas = tk.Canvas(top, bg="#2d3436")
        scroll_y = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        photo = ImageTk.PhotoImage(Image.fromarray(display_img))
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo
        canvas.configure(scrollregion=(0, 0, w, h))

    def _save_image(self, img):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Image As"
        )
        if path:
            cv2.imwrite(path, img)
            messagebox.showinfo("Success", f"High-resolution image saved successfully to:\n{path}")

    def run_segmentation(self):
        if self.original_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        self.status_var.set("Processing Segmentation Architecture...")
        self.root.update()
        
        # Ensure image is binarized for smearing
        binary_inv = cv2.adaptiveThreshold(
            self.original_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        x_stretch, y_stretch = self.x_var.get(), self.y_var.get()
        
        # ACF Analyzer (FFT Auto-Tune) Step
        if self.use_fft.get():
            try:
                from fft_analyzer import FFTAnalyzer
                results = FFTAnalyzer.calculate_optimal_parameters(self.input_path)
                if results and isinstance(results, dict):
                    x_stretch = results.get("x_stretch", x_stretch)
                    y_stretch = results.get("y_stretch", y_stretch)
                    self.x_var.set(x_stretch)
                    self.y_var.set(y_stretch)
            except Exception as e:
                print(f"[!] FFT ACF tuning skipped or failed: {e}")
        
        # Ensure odd kernel sizes for Gaussian Blur
        x_stretch = x_stretch | 1
        y_stretch = y_stretch | 1
        
        # Step 2: Gaussian Blur (Anisotropic)
        smeared = cv2.GaussianBlur(binary_inv, (x_stretch, y_stretch), 0)
        self._update_display(1, smeared)
        
        # Step 3: Thresholding (Solid Blobs)
        _, blobs = cv2.threshold(smeared, 50, 255, cv2.THRESH_BINARY)
        self._update_display(2, blobs)
        
        # Step 4: Contours
        contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = cv2.cvtColor(self.original_img.copy(), cv2.COLOR_GRAY2BGR)
        
        line_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise (width > 10% of image, height > 10)
            if w > self.original_img.shape[1] * 0.1 and h > 10:
                line_regions.append((y, y + h))
                cv2.rectangle(contours_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        self._update_display(3, contours_img)
        
        # Step 5: Mask Build (Erase non-text regions)
        mask = np.zeros_like(self.original_img)
        for sy, ey in line_regions:
            # Add a small pad
            sy = max(0, sy - 5)
            ey = min(self.original_img.shape[0], ey + 5)
            mask[sy:ey, :] = 255
            
        cleaned_mask = cv2.bitwise_and(self.original_img, self.original_img, mask=mask)
        self._update_display(4, cleaned_mask)
        
        # Step 6: Extracted Lines (Preview - Cut Separately)
        line_regions = sorted(line_regions, key=lambda r: r[0])
        
        # Calculate total height with thick padding to emphasize they are cut
        pad = 25
        total_h = sum([ey - sy for sy, ey in line_regions]) + (len(line_regions) * (pad * 2))
        total_h = max(total_h, self.original_img.shape[0])
        
        # Create a dark gray canvas so the cuts pop out
        lines_img_bgr = np.ones((total_h, self.original_img.shape[1], 3), dtype=np.uint8) * 40
        offset = pad
        
        for idx, (sy, ey) in enumerate(line_regions):
            h = ey - sy
            if offset + h < lines_img_bgr.shape[0]:
                crop = self.original_img[sy:ey, :]
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                
                # Draw a bright green border around the crop to highlight the separation
                cv2.rectangle(crop_bgr, (0, 0), (crop_bgr.shape[1]-1, crop_bgr.shape[0]-1), (0, 255, 0), 2)
                
                lines_img_bgr[offset:offset+h, :] = crop_bgr
                
                # Add text label above the cut
                cv2.putText(lines_img_bgr, f"CUT {idx+1}", (10, offset - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                offset += h + (pad * 2)
                
        self._update_display(5, lines_img_bgr)
        
        self.status_var.set(f"Extraction Complete! Found & Cut {len(line_regions)} lines (σ_x = {x_stretch}, σ_y = {y_stretch}).")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationVisualizerApp(root)
    root.mainloop()
