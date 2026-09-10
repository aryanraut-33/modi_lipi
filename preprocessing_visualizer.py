import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys

# Add paths to allow importing the existing preprocessing logic
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "preprocessing"))

try:
    from preprocess import preprocess_image_array, PreprocessConfig
except ImportError as e:
    print(f"Error importing preprocess module: {e}")

class PreprocessingVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stage 1: Preprocessing Pipeline Visualizer")
        self.root.geometry("1400x900")
        
        self.input_path = None
        self.original_img = None
        self.current_images = {}
        
        self._build_ui()
        
    def _build_ui(self):
        # ---------------- Control Panel ----------------
        control_panel = ttk.Frame(self.root, padding=10)
        control_panel.pack(fill=tk.X)
        
        ttk.Button(control_panel, text="Open Raw Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_panel, text="Run Preprocessing", command=self.run_preprocessing).pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Load a raw archaic document image to start.")
        ttk.Label(control_panel, textvariable=self.status_var, font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=20)
        
        # ---------------- Parameter Tuning ----------------
        tuning_frame = ttk.LabelFrame(control_panel, text="Adaptive Threshold Parameters")
        tuning_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(tuning_frame, text="Block Size (Odd):").grid(row=0, column=0, padx=5)
        self.block_size_var = tk.IntVar(value=61)
        tk.Scale(tuning_frame, variable=self.block_size_var, from_=3, to=151, orient=tk.HORIZONTAL, resolution=2).grid(row=0, column=1, padx=5)
        
        ttk.Label(tuning_frame, text="Constant C:").grid(row=0, column=2, padx=5)
        self.c_var = tk.IntVar(value=21)
        tk.Scale(tuning_frame, variable=self.c_var, from_=0, to=50, orient=tk.HORIZONTAL).grid(row=0, column=3, padx=5)

        # ---------------- Pipeline Visualization Grid ----------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="Preprocessing Architecture Steps")
        
        # Configure a 3x2 Grid for the 6 stages
        for i in range(3):
            self.pipeline_tab.columnconfigure(i, weight=1)
            self.pipeline_tab.rowconfigure(1, weight=1)
            self.pipeline_tab.rowconfigure(3, weight=1)
            
        labels = [
            "1. Raw Input Image", 
            "2. Grayscale Conversion", 
            "3. Bilateral Denoising",
            "4. Adaptive Binarization",
            "5. Speckle Removal (Cleaned)",
            "6. Deskewed Output (Final)"
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
        
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)
        
    def open_image(self):
        path = filedialog.askopenfilename(title="Select Raw Image")
        if not path:
            return
            
        self.input_path = path
        # Read raw image in color
        self.original_img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if self.original_img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
            
        self._update_display(0, self.original_img)
        
        # Clear previous pipeline run visuals
        for i in range(1, 6):
            self.image_labels[i].configure(image="")
            self.image_labels[i].image = None
            self.current_images.pop(i, None)
            
        self.status_var.set(f"Loaded: {os.path.basename(path)}. Ready to run preprocessing.")
        
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
        self._show_enlarged(img, f"Preprocessing Stage {index + 1} (Full Resolution)")
        
    def _show_enlarged(self, img, title="Enlarged Image"):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("1000x800")
        
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="💾 Download / Save Full Image", command=lambda: self._save_image(img)).pack(side=tk.RIGHT, padx=10)
        
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

    def run_preprocessing(self):
        if self.original_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        self.status_var.set("Processing Pipeline...")
        self.root.update()
        
        # Ensure block size is odd
        bs = self.block_size_var.get()
        if bs % 2 == 0:
            bs += 1
            self.block_size_var.set(bs)
            
        c_val = self.c_var.get()
        
        try:
            cfg = PreprocessConfig(adaptive_block_size=bs, adaptive_c=c_val)
            # return_intermediates gives us the step-by-step images based on preprocess.py
            _, intermediates = preprocess_image_array(self.original_img, cfg, return_intermediates=True)
            
            # Update displays based on intermediate dictionary returned
            self._update_display(1, intermediates.get("grayscale"))
            self._update_display(2, intermediates.get("denoised"))
            self._update_display(3, intermediates.get("binary"))
            self._update_display(4, intermediates.get("cleaned"))
            self._update_display(5, intermediates.get("deskewed"))
            
            self.status_var.set("Preprocessing Complete! Click any image to enlarge and save.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed:\n{str(e)}")
            self.status_var.set("Error during preprocessing.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingVisualizerApp(root)
    root.mainloop()
