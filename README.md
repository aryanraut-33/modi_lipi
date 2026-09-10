# Modi Lipi (ShatVidhi) Manuscript Analysis Pipeline

An end-to-end document analysis and character segmentation pipeline tailored for historical **Modi script (मोडी लिपी)** manuscripts.

The project combines domain-specific computer vision algorithms (OpenCV) with an interactive **Next.js + TypeScript** web application, allowing users to inspect intermediate stages and interactively refine character segmentation (merging and splitting glyphs) in real time.

---

## Architecture Overview

```
                         Modi Manuscript Scan
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Phase 1: Preprocessing                                          │
 │ Bilateral Filter ➔ Sauvola Binarization ➔ Cleanup ➔ HPP Deskew  │
 └────────────────────────────────┬────────────────────────────────┘
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Phase 2: Line Localisation                                      │
 │ Autocorrelation (ACF) ➔ Anisotropic Gaussian ➔ Iterative Mask   │
 └────────────────────────────────┬────────────────────────────────┘
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Phase 3: Character Segmentation                                 │
 │ Shirorekha Detection ➔ Zonal Slicing ➔ Ligature/Cluster Split   │
 └────────────────────────────────┬────────────────────────────────┘
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Interactive Web Workspace (Next.js + Flask)                     │
 │ • Real-time parameter tuning (X-stretch, Y-stretch)             │
 │ • Visual inspection of all 11 intermediate debug stages         │
 │ • Interactive Glyph Refinement (Merge Checked, Split, Reset)    │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before running the project, ensure you have the following installed on your system:

- **Python**: Version 3.9 or higher
- **Node.js**: Version 18.x or higher (LTS recommended)
- **npm**: Comes with Node.js
- **Git**: For cloning the repository

---

## Quick Start (Clone & Run)

### 1. Clone the Repository

```bash
git clone https://github.com/aryanraut-33/modi_lipi.git
cd modi_lipi
```

### 2. Set Up Python Environment

Create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

Navigate into `modi_app` and install the Node packages:

```bash
cd modi_app
npm install
cd ..
```

---

## Running the Web Application

The web interface requires **two processes**: the Flask backend bridge and the Next.js dev server. Open **two separate terminal windows or tabs**:

### Terminal 1: Start the Backend API (Flask)

Make sure your virtual environment is active, then run:

```bash
cd modi_app
python3 backend/app.py
```

* The backend will start on **`http://127.0.0.1:5005`**.
* You can test connectivity at any time:
  ```bash
  curl http://127.0.0.1:5005/api/health
  # Expected output: {"service": "Modi ShatVidhi pipeline", "status": "ok"}
  ```

### Terminal 2: Start the Web UI (Next.js)

In your second terminal:

```bash
cd modi_app
npm run dev
```

* Next.js will launch on **`http://localhost:3000`** (or **`http://localhost:3001`** if port 3000 is occupied).
* Open your browser and navigate to the displayed URL (e.g., `http://localhost:3001`).

---

## Using the Web Workspace

1. **Upload a Manuscript**:
   * Click **Choose manuscript image** in the left control panel or drag-and-drop a file.
   * Sample images are provided in the [`images_raw/`](images_raw/) directory (e.g., `example1.jpg`, `example2.jpg`).

2. **Run the Pipeline**:
   * Click **Run Modi ShatVidhi**.
   * The pipeline will process the image through all stages and display summary statistics (lines found, glyph candidates, image resolution).

3. **Inspect Intermediate Stages**:
   * **Preprocessing Tab**: View Grayscale, Bilateral Denoised, Adaptive Binary, Morphologically Cleaned, and Deskewed views. Click any card to inspect full resolution.
   * **Lines Tab**: Inspect Anisotropic Gaussian Smear, Solid Blobs, Contours, Iterative Line Mask, and Individual Extracted Line strips.

4. **Real-Time Line Tuning**:
   * Adjust **X Stretch** or **Y Stretch** sliders in the left panel.
   * Changes update **in real time** directly from memory without reloading or re-running preprocessing.
   * Toggle **ACF auto-tune** to automatically compute optimal smear parameters using autocorrelation.

5. **Interactive Glyph Refinement (Glyphs Tab)**:
   * **Merge**: Check boxes on adjacent over-segmented character fragments and click **`🔗 Merge Checked`**, or use the **`◂`** / **`▸`** buttons on any card for instant single-click neighbour merging.
   * **Split**: Click **`Split`** on any under-segmented cluster to divide it into two characters.
   * **Reset**: Click **`↺ Reset Line`** to restore the line back to the initial algorithmic output.

---

## Directory Structure

```
.
├── modi_app/                         # Next.js web application & Flask API
│   ├── backend/
│   │   └── app.py                    # Flask API bridging Python CV pipeline
│   ├── src/
│   │   ├── app/                      # Next.js App Router (pages, CSS)
│   │   ├── components/               # React components (Workspace, Controls, Glyphs)
│   │   ├── hooks/                    # API hooks (usePipeline)
│   │   └── types/                    # TypeScript interfaces
│   └── package.json
├── preprocessing/
│   └── preprocess.py                 # Bilateral filter, Sauvola binarization, HPP deskew
├── segmentation/
│   ├── line_level/                   # FFT / Autocorrelation & Anisotropic Gaussian smear
│   └── character_level/              # Shirorekha detection, Zonal slicing, Cluster splitting
├── images_raw/                       # Sample Modi manuscript scans
├── requirements.txt                  # Python dependencies
├── phase1and2.py                     # Legacy desktop Tkinter visualizer
└── README.md
```

---

## Legacy Desktop Visualizer (Optional)

If you wish to run the original Python Tkinter desktop GUI instead of the web interface:

```bash
python3 phase1and2.py
```

---

## Troubleshooting

- **Port Conflict (`Port 5005 is in use`)**:
  Kill any lingering backend processes:
  ```bash
  kill -9 $(lsof -t -i :5005)
  ```
- **Port 3000 in use by another app**:
  Next.js automatically uses `http://localhost:3001` or another available port. Check the terminal output when running `npm run dev`.
- **OpenCV Import Error**:
  Ensure your virtual environment is activated before starting `python3 backend/app.py`.
