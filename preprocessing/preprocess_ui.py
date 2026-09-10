"""
Simple desktop UI to test archaic Modi image preprocessing.

Run:
    python preprocess_ui.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from preprocess import PreprocessConfig, preprocess_image_array


class PreprocessApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modi Image Preprocessing Tester")
        self.root.geometry("1150x700")

        self.input_bgr: np.ndarray | None = None
        self.output_gray: np.ndarray | None = None
        self.input_path: str | None = None

        self._build_controls()
        self._build_canvases()

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(fill=tk.X)

        ttk.Button(panel, text="Open Image", command=self.open_image).grid(row=0, column=0, padx=6)
        ttk.Button(panel, text="Run Preprocess", command=self.run_preprocess).grid(row=0, column=1, padx=6)
        ttk.Button(panel, text="Save Output", command=self.save_output).grid(row=0, column=2, padx=6)

        ttk.Label(panel, text="Block Size").grid(row=0, column=3, padx=(24, 4))
        self.block_size_var = tk.IntVar(value=35)
        ttk.Entry(panel, textvariable=self.block_size_var, width=8).grid(row=0, column=4, padx=4)

        ttk.Label(panel, text="C").grid(row=0, column=5, padx=(16, 4))
        self.c_var = tk.IntVar(value=11)
        ttk.Entry(panel, textvariable=self.c_var, width=8).grid(row=0, column=6, padx=4)

        # Ink-based filtering is disabled in preprocess.py (kept as future scope),
        # so the UI intentionally does not expose ink-threshold controls.

        self.status_var = tk.StringVar(value="Load an image to start.")
        ttk.Label(panel, textvariable=self.status_var).grid(row=0, column=7, padx=(24, 4), sticky=tk.W)

    def _build_canvases(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        for i in range(3):
            frame.columnconfigure(i, weight=1)
        for i in range(1, 4, 2):
            frame.rowconfigure(i, weight=1)

        labels = [
            "Input", "Grayscale", "Denoised",
            "Binary Thresholding (AGT)", "Cleaned (Speckle Removal)", "Final Output"
        ]
        self.image_labels = []

        for i, text in enumerate(labels):
            row = (i // 3) * 2
            col = i % 3
            ttk.Label(frame, text=text).grid(row=row, column=col, sticky=tk.W)
            
            lbl = ttk.Label(frame)
            lbl.grid(row=row+1, column=col, sticky="nsew", padx=4, pady=4)
            self.image_labels.append(lbl)

    @staticmethod
    def _to_photo(image_rgb: np.ndarray, max_w: int = 350, max_h: int = 250) -> ImageTk.PhotoImage:
        h, w = image_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select archaic Modi document image",
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
        self.input_bgr = image
        self.output_gray = None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = self._to_photo(rgb)
        self.image_labels[0].configure(image=photo)
        self.image_labels[0].image = photo
        for i in range(1, 6):
            self.image_labels[i].configure(image="")
            self.image_labels[i].image = None
        self.status_var.set(f"Loaded: {Path(path).name}")

    def run_preprocess(self) -> None:
        if self.input_bgr is None:
            messagebox.showwarning("No Input", "Please load an image first.")
            return

        try:
            cfg = PreprocessConfig(
                adaptive_block_size=self.block_size_var.get(),
                adaptive_c=self.c_var.get(),
            )
            output, intermediates = preprocess_image_array(self.input_bgr, cfg, return_intermediates=True)
        except Exception as exc:
            messagebox.showerror("Preprocess Failed", str(exc))
            return

        self.output_gray = output
        
        stages = [
            intermediates["grayscale"],
            intermediates["denoised"],
            intermediates["binary"],
            intermediates["cleaned"],
            intermediates["ink_only"]
        ]
        
        for i, img in enumerate(stages, start=1):
            if len(img.shape) == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = img
            photo = self._to_photo(rgb)
            self.image_labels[i].configure(image=photo)
            self.image_labels[i].image = photo
            
        self.status_var.set("Preprocessing complete.")

    def save_output(self) -> None:
        if self.output_gray is None:
            messagebox.showwarning("No Output", "Run preprocessing first.")
            return

        default_name = "preprocessed.png"
        if self.input_path:
            default_name = f"{Path(self.input_path).stem}_preprocessed.png"

        path = filedialog.asksaveasfilename(
            title="Save preprocessed image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.output_gray)
        if not ok:
            messagebox.showerror("Save Failed", "Could not save the output image.")
            return

        self.status_var.set(f"Saved: {Path(path).name}")
        messagebox.showinfo("Saved", f"Output saved to:\n{path}")


def main() -> None:
    root = tk.Tk()
    app = PreprocessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
