# Modi ShatVidhi Web Workspace

This Next.js interface moves the existing desktop visualizers onto the web. It
does not reimplement the document algorithms: the Flask service calls the
existing Python preprocessing, line segmentation, FFT/ACF tuning, and character
segmentation modules in the parent project.

## Run locally

In one terminal, start the Python service:

```bash
cd "/Users/aryan/ly project/pipeline_draft1/modi_app"
python3 backend/app.py
```

In a second terminal, start the web interface:

```bash
cd "/Users/aryan/ly project/pipeline_draft1/modi_app"
npm run dev
```

Open the URL printed by Next.js (normally `http://localhost:3000` or the next
available port). The API listens on `http://127.0.0.1:5005` (or configured via `$PORT`).

## Included pipeline views

1. Upload and tuning controls.
2. Grayscale, denoising, adaptive threshold, speckle removal, and deskewing.
3. Anisotropic smear, blob mask, contours, iterative mask, and extracted lines.
4. Per-line Shirorekha/baseline overlays and normalized 64×64 glyph candidates.
5. A transparent handoff point for a future Phase 4 Modi recognition model.

The current repository does not contain trained Modi-to-Devanagari recognition
weights. The interface therefore never fabricates a transliteration; it exposes
the correct image tensors and line order for a future model endpoint.
