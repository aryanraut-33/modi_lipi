import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ACFVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ACF & HPP Signal Analyzer")
        self.root.geometry("1000x800")
        
        # ---------------- Control Panel ----------------
        control_panel = ttk.Frame(self.root, padding=10)
        control_panel.pack(fill=tk.X)
        
        ttk.Button(control_panel, text="Open Deskewed Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Load a preprocessed/deskewed image to analyze the frequency.")
        ttk.Label(control_panel, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        
        # ---------------- Matplotlib Plot Area ----------------
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure with 2 subplots (HPP on top, ACF on bottom)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial empty styling
        self.ax1.set_title("Raw Horizontal Projection Profile (HPP) - Noisy Signal")
        self.ax2.set_title("Autocorrelation Function (ACF) - Isolated Spikes")
        self.fig.tight_layout()
        self.canvas.draw()
        
    def open_image(self):
        path = filedialog.askopenfilename(title="Select Deskewed Image")
        if not path:
            return
            
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
            
        self.status_var.set(f"Analyzing {os.path.basename(path)}...")
        self.root.update()
        
        # 1. Binarize (Using exact logic from your fft_analyzer.py)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # 2. Extract HPP
        hpp = np.sum(binary, axis=1) / 255.0
        
        # 3. Calculate ACF
        hpp_centered = hpp - np.mean(hpp)
        acf = np.correlate(hpp_centered, hpp_centered, mode='full')
        acf = acf[len(acf)//2:] # Keep only positive lags
        
        # 4. Find Peaks
        peaks, properties = find_peaks(acf, prominence=0.05 * np.max(acf), distance=15)
        
        # ---------------- Plotting ----------------
        self.ax1.clear()
        self.ax2.clear()
        
        # Top Graph: HPP
        self.ax1.plot(hpp, color='#3498db', linewidth=1.5)
        self.ax1.set_title("Raw Horizontal Projection Profile (HPP) - Noisy Signal")
        self.ax1.set_ylabel("Ink Density")
        self.ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Bottom Graph: ACF
        self.ax2.plot(acf, color='#2ecc71', linewidth=2)
        self.ax2.set_title("Autocorrelation Function (ACF) - Isolated Spikes")
        self.ax2.set_xlabel("Lag Distance (Pixels)")
        self.ax2.set_ylabel("Correlation Strength")
        self.ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Mark Peaks
        if len(peaks) > 0:
            # Highlight all found peaks with Red X marks
            self.ax2.plot(peaks, acf[peaks], "rx", markersize=10, markeredgewidth=2, label="Detected Peaks")
            
            # The first major peak is Period P
            period = peaks[0]
            self.ax2.axvline(x=period, color='#e74c3c', linestyle='--', alpha=0.8, label=f"Period P = {period}px")
            
            self.status_var.set(f"Analysis Complete! Dominant Line Gap (Period P) = {period} pixels.")
            self.ax2.legend()
        else:
            self.status_var.set("Analysis Complete. No strong periodic peaks found (Are there lines?).")
            
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ACFVisualizerApp(root)
    root.mainloop()
