"""
Simple desktop UI to test the LineTR text line segmentation pipeline.

Run:
    python linetr_ui.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import the inference pipeline orchestrator
from main import run_inference


class LineTRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LineTR Segmentation Tester")
        self.root.geometry("1100x650")

        self.input_path: str | None = None
        self.input_rgb: np.ndarray | None = None
        self.output_rgb: np.ndarray | None = None
        self.polygons: list | None = None

        self._build_controls()
        self._build_canvases()

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(fill=tk.X)

        ttk.Button(panel, text="Open Image", command=self.open_image).grid(row=0, column=0, padx=6)
        ttk.Button(panel, text="Run Segmentation", command=self.run_segmentation).grid(row=0, column=1, padx=6)
        ttk.Button(panel, text="Save Output", command=self.save_output).grid(row=0, column=2, padx=6)

        ttk.Label(panel, text="Config Path:").grid(row=0, column=3, padx=(24, 4))
        self.config_var = tk.StringVar(value="config.yaml")
        ttk.Entry(panel, textvariable=self.config_var, width=15).grid(row=0, column=4, padx=4)

        ttk.Label(panel, text="Weights Path (Optional):").grid(row=0, column=5, padx=(16, 4))
        self.weights_var = tk.StringVar(value="")
        ttk.Entry(panel, textvariable=self.weights_var, width=20).grid(row=0, column=6, padx=4)

        self.status_var = tk.StringVar(value="Load a manuscript document image to start.")
        ttk.Label(panel, textvariable=self.status_var).grid(row=1, column=0, columnspan=7, pady=10, sticky=tk.W)

    def _build_canvases(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Input Document").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(frame, text="Segmented Output").grid(row=0, column=1, sticky=tk.W)
        
        self.lbl_input = ttk.Label(frame)
        self.lbl_input.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        
        self.lbl_output = ttk.Label(frame)
        self.lbl_output.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

    @staticmethod
    def _to_photo(image_rgb: np.ndarray, max_w: int = 500, max_h: int = 500) -> ImageTk.PhotoImage:
        h, w = image_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select manuscript document image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Error", "Could not open selected image.")
            return

        self.input_path = path
        self.input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.output_rgb = None
        self.polygons = None

        photo = self._to_photo(self.input_rgb)
        self.lbl_input.configure(image=photo)
        self.lbl_input.image = photo
        
        self.lbl_output.configure(image="")
        self.lbl_output.image = None
        
        self.status_var.set(f"Loaded: {Path(path).name}")

    def run_segmentation(self) -> None:
        if self.input_path is None:
            messagebox.showwarning("No Input", "Please load an image first.")
            return

        self.status_var.set("Running LineTR Segmentation... This may take a moment.")
        self.root.update()

        try:
            weights_path = self.weights_var.get().strip()
            if not weights_path:
                weights_path = None
                
            config_path = self.config_var.get().strip()
            
            # Execute the End-to-End LineTR Inference Pipeline
            self.polygons = run_inference(self.input_path, model_weights_path=weights_path, config_path=config_path)
            
            # Draw polygons on a copy of the input image
            output_image = self.input_rgb.copy()
            
            # Generate random distinct colors for different lines
            for i, polygon in enumerate(self.polygons):
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Pick a random bright color for visualization
                color = np.random.randint(50, 255, size=(3,)).tolist()
                
                # Draw the polygonal boundary
                cv2.polylines(output_image, [pts], isClosed=True, color=color, thickness=3)
                
            self.output_rgb = output_image
            
            photo = self._to_photo(self.output_rgb)
            self.lbl_output.configure(image=photo)
            self.lbl_output.image = photo
            
            self.status_var.set(f"Segmentation complete. Found {len(self.polygons)} tight-fitting line polygons.")
            
        except Exception as exc:
            messagebox.showerror("Segmentation Failed", str(exc))
            self.status_var.set("Segmentation failed. Check console for details.")
            raise exc # Re-raise for console debugging

    def save_output(self) -> None:
        if self.output_rgb is None:
            messagebox.showwarning("No Output", "Run segmentation first.")
            return

        default_name = "segmented.png"
        if self.input_path:
            default_name = f"{Path(self.input_path).stem}_segmented.png"

        path = filedialog.asksaveasfilename(
            title="Save segmented image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return

        # Convert back to BGR for saving via cv2
        bgr_output = cv2.cvtColor(self.output_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, bgr_output)
        if not ok:
            messagebox.showerror("Save Failed", "Could not save the output image.")
            return

        self.status_var.set(f"Saved: {Path(path).name}")
        messagebox.showinfo("Saved", f"Output saved to:\n{path}")


def main() -> None:
    root = tk.Tk()
    app = LineTRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
