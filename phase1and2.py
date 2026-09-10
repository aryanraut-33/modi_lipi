import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image, ImageTk

# --- Path Injections ---
ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT, "preprocessing"))
sys.path.append(os.path.join(ROOT, "segmentation", "line_level", "implementation", "Image Processing", "Suite"))
sys.path.append(os.path.join(ROOT, "segmentation", "character_level"))

from preprocess import preprocess_image_array, PreprocessConfig
from segmentation_engine import AnisotropicGaussianSegmenter
from fft_analyzer import FFTAnalyzer
from character_segmenter import CharacterSegmenter, CharSegConfig, SegmentedGlyph

class ScrollableFrame(ttk.Frame):
    """A reusable vertically scrollable frame widget."""
    def __init__(container, *args, **kwargs):
        super(ScrollableFrame, container).__init__(*args, **kwargs)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        container.scrollable_frame = ttk.Frame(canvas)

        container.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=container.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)


class EndToEndApp:
    """
    Modi ShatVidhi Desktop Research Application:
    - Step 1: Preprocessing & Line-Level Segmentation (Phases 1 & 2) -> Review in Tab 2.
    - Step 2: Algorithmic Character-Level Segmentation (Phase 3) -> Review in Tab 3.
    - Interactive Refinement: Merge adjacent over-segmented glyphs or split under-segmented clusters with 1 click.
    """
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modi ShatVidhi: Archaic Modi Transliteration Pipeline (Phases 1-3)")
        self.root.geometry("1420x940")

        self.input_bgr = None
        self.clean_gray = None
        self.input_path = None
        self.final_line_regions = []
        self.final_document_glyphs = []      # List[List[SegmentedGlyph]]
        self.raw_document_glyphs_backup = [] # Backup for reset
        self.line_annotated_images = []      # RGB Annotated Line Overviews
        
        self.temp_preprocessed = "temp_preprocessed.png"
        self.temp_loop = "temp_loop_mask.png"

        # Character Segmenter Engine
        self.char_segmenter = CharacterSegmenter()

        # Selection state for multi-merge in Tab 3
        self.selected_glyph_indices = set()
        self.glyph_check_vars = []

        self._build_controls()
        self._build_tabs()

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(fill=tk.X)

        # ROW 0: Primary Execution Controls
        self.btn_open = ttk.Button(panel, text="Open Image", command=self.open_image)
        self.btn_open.grid(row=0, column=0, padx=4, pady=4)
        
        self.btn_run_lines = ttk.Button(panel, text="Step 1: Run Preprocessing & Lines (Phases 1 & 2)", command=self.run_phase1_and_2)
        self.btn_run_lines.grid(row=0, column=1, padx=4, pady=4)

        self.btn_run_chars = ttk.Button(panel, text="Step 2: Proceed to Character Segmentation (Phase 3) >>", command=self.run_phase3, state=tk.DISABLED)
        self.btn_run_chars.grid(row=0, column=2, padx=6, pady=4)

        ttk.Separator(panel, orient='vertical').grid(row=0, column=3, sticky='ns', padx=8)

        ttk.Label(panel, text="Block Size:").grid(row=0, column=4, padx=2)
        self.block_size_var = tk.IntVar(value=35)
        ttk.Entry(panel, textvariable=self.block_size_var, width=4).grid(row=0, column=5, padx=2)

        ttk.Label(panel, text="Adaptive C:").grid(row=0, column=6, padx=(6, 2))
        self.c_var = tk.IntVar(value=11)
        ttk.Entry(panel, textvariable=self.c_var, width=4).grid(row=0, column=7, padx=2)

        self.status_var = tk.StringVar(value="Load an image and click Step 1.")
        ttk.Label(panel, textvariable=self.status_var, font=("Arial", 9, "italic"), foreground="#0066cc").grid(row=0, column=8, padx=10, sticky=tk.W)

        # ROW 1: Phase 2 Line Tuning Controls
        ttk.Separator(panel, orient='horizontal').grid(row=1, column=0, columnspan=12, sticky='ew', pady=6)
        
        self.use_fft_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="FFT Auto-Tune", variable=self.use_fft_var).grid(row=2, column=0, sticky=tk.W, padx=4)
        
        ttk.Label(panel, text="FFT Iter:").grid(row=2, column=1, padx=2)
        self.iter_var = tk.IntVar(value=3)
        tk.Scale(panel, variable=self.iter_var, from_=1, to=8, resolution=1, orient=tk.HORIZONTAL, showvalue=True, length=70).grid(row=2, column=2, padx=2)
        
        ttk.Label(panel, text="X-Stretch:").grid(row=2, column=3, padx=(6, 2))
        self.x_var = tk.IntVar(value=41)
        sc_x = tk.Scale(panel, variable=self.x_var, from_=11, to=301, resolution=2, orient=tk.HORIZONTAL, showvalue=True, length=80)
        sc_x.grid(row=2, column=4, padx=2)
        sc_x.bind("<ButtonRelease-1>", self._quick_segment_update)
        
        ttk.Label(panel, text="Y-Stretch:").grid(row=2, column=5, padx=(6, 2))
        self.y_var = tk.IntVar(value=5)
        sc_y = tk.Scale(panel, variable=self.y_var, from_=3, to=51, resolution=2, orient=tk.HORIZONTAL, showvalue=True, length=70)
        sc_y.grid(row=2, column=6, padx=2)
        sc_y.bind("<ButtonRelease-1>", self._quick_segment_update)
        
        ttk.Label(panel, text="Pad:").grid(row=2, column=7, padx=(6, 2))
        self.pad_var = tk.IntVar(value=5)
        sc_pad = tk.Scale(panel, variable=self.pad_var, from_=0, to=20, resolution=1, orient=tk.HORIZONTAL, showvalue=True, length=60)
        sc_pad.grid(row=2, column=8, padx=2)
        sc_pad.bind("<ButtonRelease-1>", self._quick_segment_update)

        # Phase 3 Cluster Control
        ttk.Separator(panel, orient='vertical').grid(row=2, column=9, sticky='ns', padx=6)
        ttk.Label(panel, text="Cluster Split:").grid(row=2, column=10, padx=2)
        self.cluster_split_var = tk.DoubleVar(value=1.60)
        sc_cs = tk.Scale(panel, variable=self.cluster_split_var, from_=1.1, to=2.5, resolution=0.05, orient=tk.HORIZONTAL, showvalue=True, length=70)
        sc_cs.grid(row=2, column=11, padx=2)

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # TAB 1: PREPROCESSING (2x3 Grid)
        self.tab_preproc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_preproc, text="1. Preprocessing Stages")
        
        for i in range(3):
            self.tab_preproc.columnconfigure(i, weight=1)
        for i in range(1, 4, 2):
            self.tab_preproc.rowconfigure(i, weight=1)

        labels = [
            "1. RAW Input", "2. Grayscale", "3. Denoised",
            "4. Binary Thresholding", "5. Cleaned (Speckle Removal)", "6. Deskewed Output"
        ]
        self.image_labels = []

        for i, text in enumerate(labels):
            row = (i // 3) * 2
            col = i % 3
            ttk.Label(self.tab_preproc, text=text, font=("Arial", 10, "bold")).grid(row=row, column=col, sticky=tk.W, padx=6, pady=(6, 0))
            
            lbl = ttk.Label(self.tab_preproc)
            lbl.grid(row=row+1, column=col, sticky="nsew", padx=6, pady=4)
            self.image_labels.append(lbl)

        # TAB 2: LINE SEGMENTATION (Extracted Lines)
        self.tab_seg = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_seg, text="2. Extracted Lines (Review & Adjust)")
        
        seg_top_bar = ttk.Frame(self.tab_seg, padding=6)
        seg_top_bar.pack(fill=tk.X)
        ttk.Label(seg_top_bar, text="Review extracted text lines below. Tune sliders above if needed.", font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=6)
        
        self.btn_tab2_proceed = ttk.Button(seg_top_bar, text=">> Looks Good! Proceed to Character Segmentation (Phase 3)", command=self.run_phase3, state=tk.DISABLED)
        self.btn_tab2_proceed.pack(side=tk.RIGHT, padx=6)

        self.scroll_view_lines = ScrollableFrame(self.tab_seg)
        self.scroll_view_lines.pack(fill=tk.BOTH, expand=True)

        # TAB 3: CHARACTER GLYPH SEGMENTATION (Interactive Merge & Split)
        self.tab_char_seg = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_char_seg, text="3. Character Glyphs (Interactive Refinement)")
        
        char_top_bar = ttk.Frame(self.tab_char_seg, padding=8)
        char_top_bar.pack(fill=tk.X)

        ttk.Label(char_top_bar, text="Select Line:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=6)
        self.line_selector_var = tk.StringVar()
        self.combo_lines = ttk.Combobox(char_top_bar, textvariable=self.line_selector_var, state="readonly", width=28)
        self.combo_lines.pack(side=tk.LEFT, padx=6)
        self.combo_lines.bind("<<ComboboxSelected>>", self._on_line_selected_for_glyphs)

        ttk.Separator(char_top_bar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=8)

        # Action Buttons
        self.btn_merge_selected = ttk.Button(char_top_bar, text="🔗 Merge Checked Glyphs", command=self.merge_checked_glyphs)
        self.btn_merge_selected.pack(side=tk.LEFT, padx=4)

        self.btn_reset_line = ttk.Button(char_top_bar, text="↺ Reset Line", command=self.reset_current_line)
        self.btn_reset_line.pack(side=tk.LEFT, padx=4)

        self.char_summary_var = tk.StringVar(value="Select adjacent glyphs to merge or click '< Merge' / 'Split' on any card.")
        ttk.Label(char_top_bar, textvariable=self.char_summary_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=12)

        self.scroll_view_glyphs = ScrollableFrame(self.tab_char_seg)
        self.scroll_view_glyphs.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _to_photo(image_rgb: np.ndarray, max_w: int = 350, max_h: int = 250) -> ImageTk.PhotoImage:
        h, w = image_rgb.shape[:2]
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select archaic Modi document image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Error", "Could not open selected image.")
            return

        self.input_path = path
        self.input_bgr = image
        self.clean_gray = None
        self.final_line_regions = []
        self.final_document_glyphs = []
        self.raw_document_glyphs_backup = []
        self.line_annotated_images = []

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = self._to_photo(rgb)
        self.image_labels[0].configure(image=photo)
        self.image_labels[0].image = photo
        
        for i in range(1, 6):
            self.image_labels[i].configure(image="")
            self.image_labels[i].image = None
            
        for widget in self.scroll_view_lines.scrollable_frame.winfo_children():
            widget.destroy()

        for widget in self.scroll_view_glyphs.scrollable_frame.winfo_children():
            widget.destroy()
        self.combo_lines['values'] = []
        self.line_selector_var.set("")
        self.btn_run_chars.config(state=tk.DISABLED)
        self.btn_tab2_proceed.config(state=tk.DISABLED)
        self.char_summary_var.set("Image loaded. Click 'Step 1: Run Preprocessing & Lines'.")

        self.status_var.set(f"Loaded: {Path(path).name} -> Click 'Step 1' to extract lines.")

    # -------------------------------------------------------------
    # STEP 1: PREPROCESSING & LINE SEGMENTATION (FAST & LOCAL)
    # -------------------------------------------------------------
    def run_phase1_and_2(self) -> None:
        if self.input_bgr is None:
            messagebox.showwarning("No Input", "Please load an image first.")
            return

        self.btn_run_lines.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_run_chars.config(state=tk.DISABLED)
        self.btn_tab2_proceed.config(state=tk.DISABLED)
        self.status_var.set("Running Phase 1 & Phase 2...")
        
        threading.Thread(target=self._process_phase1_and_2_thread, daemon=True).start()

    def _process_phase1_and_2_thread(self):
        try:
            cfg = PreprocessConfig(
                adaptive_block_size=self.block_size_var.get(),
                adaptive_c=self.c_var.get(),
            )
            clean_img, intermediates = preprocess_image_array(self.input_bgr, cfg, return_intermediates=True)
            self.clean_gray = clean_img
            cv2.imwrite(self.temp_preprocessed, clean_img)
            
            segmenter = AnisotropicGaussianSegmenter()
            pad = self.pad_var.get()
            line_regions = []
            
            if self.use_fft_var.get():
                shutil.copy(self.temp_preprocessed, self.temp_loop)
                current_x, current_y = self.x_var.get(), self.y_var.get()
                iterations = self.iter_var.get()
                
                for _ in range(iterations):
                    fft_results = FFTAnalyzer.calculate_optimal_parameters(self.temp_loop)
                    if fft_results:
                        current_x = fft_results.get("x_stretch", current_x)
                        current_y = fft_results.get("y_stretch", current_y)
                    
                    _, _, line_regions, _ = segmenter.segment(
                        self.temp_preprocessed, x_stretch=current_x, y_stretch=current_y, pad=pad
                    )
                    
                    masked_img = np.ones_like(clean_img) * 255
                    for start_y, end_y in line_regions:
                        sy, ey = max(0, start_y), min(clean_img.shape[0], end_y)
                        masked_img[sy:ey, :] = clean_img[sy:ey, :]
                    cv2.imwrite(self.temp_loop, masked_img)
                
                self.root.after(0, lambda x=current_x, y=current_y: (self.x_var.set(x), self.y_var.set(y)))
            else:
                current_x, current_y = self.x_var.get(), self.y_var.get()
                _, _, line_regions, _ = segmenter.segment(
                    self.temp_preprocessed, x_stretch=current_x, y_stretch=current_y, pad=pad
                )
            
            self.final_line_regions = line_regions
            self.root.after(0, self._render_phase1_and_2_ui, intermediates, clean_img)

        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _render_phase1_and_2_ui(self, intermediates, clean_img):
        # Update Tab 1
        stages = [
            intermediates["grayscale"],
            intermediates["denoised"],
            intermediates["binary"],
            intermediates["cleaned"],
            intermediates["deskewed"]
        ]
        for i, img in enumerate(stages, start=1):
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
            photo = self._to_photo(rgb)
            self.image_labels[i].configure(image=photo)
            self.image_labels[i].image = photo

        # Update Tab 2
        for widget in self.scroll_view_lines.scrollable_frame.winfo_children():
            widget.destroy()
            
        header = ttk.Label(self.scroll_view_lines.scrollable_frame, 
                           text=f"Extracted Text Lines ({len(self.final_line_regions)})", 
                           font=("Arial", 14, "bold"))
        header.pack(anchor="w", pady=(10, 5), padx=10)
        
        for idx, (start_y, end_y) in enumerate(self.final_line_regions):
            sy, ey = max(0, start_y), min(clean_img.shape[0], end_y)
            line_crop = clean_img[sy:ey, :]
            photo = self._to_photo(cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB), max_w=1100, max_h=250)
            
            frame = ttk.Frame(self.scroll_view_lines.scrollable_frame, padding=5)
            frame.pack(fill=tk.X, padx=10, pady=4)
            
            img_lbl = ttk.Label(frame, image=photo)
            img_lbl.image = photo
            img_lbl.pack(side=tk.TOP, anchor="w")
            
            txt_lbl = ttk.Label(frame, text=f"Line {idx + 1}  |  Vertical Span: [y={sy} to y={ey}]")
            txt_lbl.pack(side=tk.TOP, anchor="w", pady=(2, 0))
            ttk.Separator(self.scroll_view_lines.scrollable_frame, orient='horizontal').pack(fill='x', padx=10, pady=4)

        self.btn_run_lines.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self.btn_run_chars.config(state=tk.NORMAL)
        self.btn_tab2_proceed.config(state=tk.NORMAL)
        self.status_var.set(f"Extracted {len(self.final_line_regions)} lines. Review in Tab 2, then click Step 2.")
        self.notebook.select(self.tab_seg)

    # -------------------------------------------------------------
    # STEP 2: FAST ALGORITHMIC CHARACTER SEGMENTATION (100% LOCAL)
    # -------------------------------------------------------------
    def run_phase3(self) -> None:
        if not self.final_line_regions or self.clean_gray is None:
            messagebox.showwarning("Lines Missing", "Please run Step 1 (Line Segmentation) first.")
            return

        self.btn_run_lines.config(state=tk.DISABLED)
        self.btn_run_chars.config(state=tk.DISABLED)
        self.btn_tab2_proceed.config(state=tk.DISABLED)
        self.status_var.set(f"Extracting character glyphs across {len(self.final_line_regions)} lines...")

        threading.Thread(target=self._process_phase3_thread, daemon=True).start()

    def _process_phase3_thread(self):
        try:
            clean_img = self.clean_gray
            line_regions = self.final_line_regions
            
            self.char_segmenter.cfg.cluster_width_multiplier = self.cluster_split_var.get()
            self.final_document_glyphs = []
            self.raw_document_glyphs_backup = []
            self.line_annotated_images = []

            for line_idx, (sy, ey) in enumerate(line_regions):
                sy_clamped, ey_clamped = max(0, sy), min(clean_img.shape[0], ey)
                line_crop = clean_img[sy_clamped:ey_clamped, :]
                
                glyphs, debug_info = self.char_segmenter.segment_line(line_crop, line_id=line_idx)
                self.final_document_glyphs.append(list(glyphs))
                self.raw_document_glyphs_backup.append(list(glyphs))
                self._update_line_annotated_image(line_idx)

            self.root.after(0, self._render_phase3_ui)

        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _update_line_annotated_image(self, line_idx: int):
        sy, ey = self.final_line_regions[line_idx]
        sy_clamped, ey_clamped = max(0, sy), min(self.clean_gray.shape[0], ey)
        line_crop = self.clean_gray[sy_clamped:ey_clamped, :]
        glyphs = self.final_document_glyphs[line_idx]

        annotated_rgb = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB)
        h_line, w_line = line_crop.shape[:2]
        ls = glyphs[0].metadata.get("shirorekha_y", int(h_line * 0.35)) if glyphs else int(h_line * 0.35)
        lb = glyphs[0].metadata.get("baseline_y", int(h_line * 0.75)) if glyphs else int(h_line * 0.75)
        
        cv2.line(annotated_rgb, (0, ls), (w_line, ls), (0, 180, 255), 1, cv2.LINE_AA)
        cv2.line(annotated_rgb, (0, lb), (w_line, lb), (255, 120, 0), 1, cv2.LINE_AA)
        
        for g in glyphs:
            x1, y1, x2, y2 = g.bbox
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(annotated_rgb, str(g.glyph_id + 1), (x1 + 2, min(y2 - 2, ls + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 0, 0), 1)

        if line_idx < len(self.line_annotated_images):
            self.line_annotated_images[line_idx] = annotated_rgb
        else:
            self.line_annotated_images.append(annotated_rgb)

    def _render_phase3_ui(self):
        self._update_character_tab()
        self.btn_run_lines.config(state=tk.NORMAL)
        self.btn_run_chars.config(state=tk.NORMAL)
        self.btn_tab2_proceed.config(state=tk.NORMAL)
        self.status_var.set("Phase 3 complete! Click '< Merge' or 'Split' on any card to refine boundaries.")
        self.notebook.select(self.tab_char_seg)

    # -------------------------------------------------------------
    # INTERACTIVE MERGE & SPLIT REFINEMENT (1-CLICK USER ACTIONS)
    # -------------------------------------------------------------
    def merge_with_left(self, line_idx: int, glyph_idx: int):
        if glyph_idx <= 0: return
        self.merge_pair_of_glyphs(line_idx, glyph_idx - 1, glyph_idx)

    def merge_with_right(self, line_idx: int, glyph_idx: int):
        if glyph_idx >= len(self.final_document_glyphs[line_idx]) - 1: return
        self.merge_pair_of_glyphs(line_idx, glyph_idx, glyph_idx + 1)

    def merge_pair_of_glyphs(self, line_idx: int, idx1: int, idx2: int):
        sy, ey = self.final_line_regions[line_idx]
        sy_clamped, ey_clamped = max(0, sy), min(self.clean_gray.shape[0], ey)
        line_crop = self.clean_gray[sy_clamped:ey_clamped, :]
        current_glyphs = self.final_document_glyphs[line_idx]

        updated = self.char_segmenter.merge_glyph_spans(
            line_crop, current_glyphs, [idx1, idx2], line_id=line_idx
        )
        self.final_document_glyphs[line_idx] = updated
        self._update_line_annotated_image(line_idx)
        self._refresh_character_view(line_idx)
        self.status_var.set(f"Merged glyph #{idx1+1} and #{idx2+1} into a single character!")

    def merge_checked_glyphs(self):
        line_idx = self.combo_lines.current()
        if line_idx < 0 or not self.final_document_glyphs: return

        selected = [i for i, v in enumerate(self.glyph_check_vars) if v.get()]
        if len(selected) < 2:
            messagebox.showinfo("Select Glyphs", "Please check at least 2 adjacent glyph cards to merge.")
            return

        sy, ey = self.final_line_regions[line_idx]
        sy_clamped, ey_clamped = max(0, sy), min(self.clean_gray.shape[0], ey)
        line_crop = self.clean_gray[sy_clamped:ey_clamped, :]
        current_glyphs = self.final_document_glyphs[line_idx]

        updated = self.char_segmenter.merge_glyph_spans(
            line_crop, current_glyphs, selected, line_id=line_idx
        )
        self.final_document_glyphs[line_idx] = updated
        self._update_line_annotated_image(line_idx)
        self._refresh_character_view(line_idx)
        self.status_var.set(f"Merged {len(selected)} selected glyphs into a complete character unit!")

    def split_target_glyph(self, line_idx: int, glyph_idx: int):
        sy, ey = self.final_line_regions[line_idx]
        sy_clamped, ey_clamped = max(0, sy), min(self.clean_gray.shape[0], ey)
        line_crop = self.clean_gray[sy_clamped:ey_clamped, :]
        current_glyphs = self.final_document_glyphs[line_idx]

        updated = self.char_segmenter.split_glyph_span(
            line_crop, current_glyphs, glyph_idx, split_ratio=0.50, line_id=line_idx
        )
        self.final_document_glyphs[line_idx] = updated
        self._update_line_annotated_image(line_idx)
        self._refresh_character_view(line_idx)
        self.status_var.set(f"Split glyph #{glyph_idx+1} into 2 individual characters!")

    def reset_current_line(self):
        line_idx = self.combo_lines.current()
        if line_idx < 0 or line_idx >= len(self.raw_document_glyphs_backup): return

        self.final_document_glyphs[line_idx] = list(self.raw_document_glyphs_backup[line_idx])
        self._update_line_annotated_image(line_idx)
        self._refresh_character_view(line_idx)
        self.status_var.set(f"Reset Line {line_idx+1} to initial segmentation.")

    def _refresh_character_view(self, line_idx: int):
        self._update_character_tab()
        self.combo_lines.current(line_idx)
        self._render_glyphs_for_line(line_idx)

    def _quick_segment_update(self, event=None):
        if self.clean_gray is None: return
        self.use_fft_var.set(False)
        self.status_var.set("Re-segmenting lines...")
        threading.Thread(target=self._quick_line_segment_thread, daemon=True).start()

    def _quick_line_segment_thread(self):
        try:
            segmenter = AnisotropicGaussianSegmenter()
            _, _, line_regions, _ = segmenter.segment(
                self.temp_preprocessed, x_stretch=self.x_var.get(), y_stretch=self.y_var.get(), pad=self.pad_var.get()
            )
            self.final_line_regions = line_regions

            def _update_ui():
                for widget in self.scroll_view_lines.scrollable_frame.winfo_children():
                    widget.destroy()
                
                header = ttk.Label(self.scroll_view_lines.scrollable_frame, 
                                   text=f"Extracted Text Lines ({len(self.final_line_regions)})", 
                                   font=("Arial", 14, "bold"))
                header.pack(anchor="w", pady=(10, 5), padx=10)
                
                for idx, (start_y, end_y) in enumerate(self.final_line_regions):
                    sy, ey = max(0, start_y), min(self.clean_gray.shape[0], end_y)
                    line_crop = self.clean_gray[sy:ey, :]
                    photo = self._to_photo(cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB), max_w=1100, max_h=250)
                    
                    frame = ttk.Frame(self.scroll_view_lines.scrollable_frame, padding=5)
                    frame.pack(fill=tk.X, padx=10, pady=4)
                    
                    img_lbl = ttk.Label(frame, image=photo)
                    img_lbl.image = photo
                    img_lbl.pack(side=tk.TOP, anchor="w")
                    
                    txt_lbl = ttk.Label(frame, text=f"Line {idx + 1}  |  Vertical Span: [y={sy} to y={ey}]")
                    txt_lbl.pack(side=tk.TOP, anchor="w", pady=(2, 0))
                    ttk.Separator(self.scroll_view_lines.scrollable_frame, orient='horizontal').pack(fill='x', padx=10, pady=4)

                self.btn_run_chars.config(state=tk.NORMAL)
                self.btn_tab2_proceed.config(state=tk.NORMAL)
                self.status_var.set(f"Lines updated ({len(self.final_line_regions)} lines). Click Step 2 when ready.")

            self.root.after(0, _update_ui)
        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _handle_error(self, err_msg):
        messagebox.showerror("Pipeline Error", err_msg)
        self.btn_run_lines.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self.status_var.set("Pipeline Error.")

    def _update_character_tab(self):
        combo_options = []
        total_glyphs = 0
        for i, glyph_list in enumerate(self.final_document_glyphs):
            count = len(glyph_list)
            total_glyphs += count
            combo_options.append(f"Line {i + 1} ({count} complete glyphs)")

        self.combo_lines['values'] = combo_options
        self.char_summary_var.set(f"Lines: {len(self.final_document_glyphs)}  |  Total Glyphs: {total_glyphs}")

        if combo_options:
            if self.combo_lines.current() < 0:
                self.combo_lines.current(0)
            curr = max(0, self.combo_lines.current())
            self._render_glyphs_for_line(curr)
        else:
            for widget in self.scroll_view_glyphs.scrollable_frame.winfo_children():
                widget.destroy()

    def _on_line_selected_for_glyphs(self, event=None):
        idx = self.combo_lines.current()
        if idx >= 0:
            self._render_glyphs_for_line(idx)

    def _render_glyphs_for_line(self, line_idx: int):
        for widget in self.scroll_view_glyphs.scrollable_frame.winfo_children():
            widget.destroy()

        if line_idx >= len(self.final_document_glyphs):
            return

        glyphs = self.final_document_glyphs[line_idx]
        annotated_line = self.line_annotated_images[line_idx] if line_idx < len(self.line_annotated_images) else None

        parent = self.scroll_view_glyphs.scrollable_frame

        # 1. Annotated Line Banner
        line_card = ttk.LabelFrame(parent, text=f"Line {line_idx + 1} Overview (Top Cyan: Shirorekha | Bottom Orange: Baseline | Green: Complete Glyphs)", padding=8)
        line_card.pack(fill=tk.X, padx=10, pady=(6, 12))

        if annotated_line is not None:
            photo_ann = self._to_photo(annotated_line, max_w=1150, max_h=200)
            lbl_ann = ttk.Label(line_card, image=photo_ann)
            lbl_ann.image = photo_ann
            lbl_ann.pack(anchor="w")

        # 2. Grid of 64x64 Normalized Complete Glyphs with Individual Interactive Actions
        grid_frame = ttk.LabelFrame(parent, text=f"Extracted Character Glyphs ({len(glyphs)} items) — Check boxes to merge multiple, or use '< Merge' / 'Split' buttons", padding=10)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.glyph_check_vars = []
        max_cols = 8

        for idx, g in enumerate(glyphs):
            row = (idx // max_cols) * 2
            col = idx % max_cols

            tensor_64 = g.to_fixed_size((64, 64))
            rgb_tensor = cv2.cvtColor(tensor_64, cv2.COLOR_GRAY2RGB)
            
            pil_img = Image.fromarray(cv2.resize(rgb_tensor, (80, 80), interpolation=cv2.INTER_NEAREST))
            photo = ImageTk.PhotoImage(pil_img)

            card = ttk.Frame(grid_frame, borderwidth=2, relief="groove", padding=6)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            # Checkbox on top
            chk_var = tk.BooleanVar(value=False)
            self.glyph_check_vars.append(chk_var)
            chk = ttk.Checkbutton(card, text=f"#{g.glyph_id + 1}", variable=chk_var)
            chk.pack(side=tk.TOP, anchor="w")

            img_lbl = ttk.Label(card, image=photo)
            img_lbl.image = photo
            img_lbl.pack(side=tk.TOP, pady=2)

            info_text = f"{g.bbox[2]-g.bbox[0]}x{g.bbox[3]-g.bbox[1]}"
            if g.has_upper_modifier:
                info_text += " | +Top"
            if g.has_lower_modifier:
                info_text += " | +Bot"

            lbl_info = ttk.Label(card, text=info_text, font=("Arial", 8), justify=tk.CENTER)
            lbl_info.pack(side=tk.TOP, pady=(1, 2))

            # Action Buttons Panel
            btn_row = ttk.Frame(card)
            btn_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

            if idx > 0:
                btn_left = ttk.Button(btn_row, text="<", width=2, command=lambda l=line_idx, g_i=idx: self.merge_with_left(l, g_i))
                btn_left.pack(side=tk.LEFT, padx=1)

            btn_split = ttk.Button(btn_row, text="Split", width=4, command=lambda l=line_idx, g_i=idx: self.split_target_glyph(l, g_i))
            btn_split.pack(side=tk.LEFT, padx=1, expand=True)

            if idx < len(glyphs) - 1:
                btn_right = ttk.Button(btn_row, text=">", width=2, command=lambda l=line_idx, g_i=idx: self.merge_with_right(l, g_i))
                btn_right.pack(side=tk.LEFT, padx=1)


def main() -> None:
    root = tk.Tk()
    app = EndToEndApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
