"""Web API bridge for the Modi ShatVidhi image-analysis pipeline.

This service deliberately calls the project's existing OpenCV pipeline rather
than duplicating the processing rules in the web application.

Session-based state management allows interactive merge/split/reset operations
on character glyphs without re-running the full pipeline.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(PROJECT_ROOT / "preprocessing"),
    str(PROJECT_ROOT / "segmentation" / "line_level" / "implementation" / "Image Processing" / "Suite"),
    str(PROJECT_ROOT / "segmentation" / "character_level"),
]

from preprocess import PreprocessConfig, preprocess_image_array  # noqa: E402
from fft_analyzer import FFTAnalyzer  # noqa: E402
from character_segmenter import CharacterSegmenter, CharSegConfig, SegmentedGlyph  # noqa: E402

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# Each session holds the processed document state so merge/split/reset
# can operate on the stored data without re-uploading.
sessions: Dict[str, Dict[str, Any]] = {}


def _get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve a session or raise 404."""
    if session_id not in sessions:
        raise KeyError(f"Session '{session_id}' not found. Please re-process the document.")
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def encode_image(image: np.ndarray) -> str:
    """Return a PNG data URL suitable for direct use in an img element."""
    if image is None or image.size == 0:
        return ""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode pipeline image")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def int_value(name: str, default: int, low: int, high: int) -> int:
    value = int(request.form.get(name, default))
    return max(low, min(high, value))


def _annotate_line(clean: np.ndarray, line_region: Tuple[int, int], glyphs: List[SegmentedGlyph]) -> np.ndarray:
    """Produce an annotated RGB image for a single line with bounding boxes."""
    sy, ey = line_region
    sy_c, ey_c = max(0, sy), min(clean.shape[0], ey)
    line_crop = clean[sy_c:ey_c, :]
    annotated = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2BGR)
    h_line, w_line = line_crop.shape[:2]

    shirorekha = glyphs[0].metadata.get("shirorekha_y", int(h_line * 0.35)) if glyphs else int(h_line * 0.35)
    baseline = glyphs[0].metadata.get("baseline_y", int(h_line * 0.75)) if glyphs else int(h_line * 0.75)

    cv2.line(annotated, (0, shirorekha), (w_line, shirorekha), (30, 161, 220), 1, cv2.LINE_AA)
    cv2.line(annotated, (0, baseline), (w_line, baseline), (29, 113, 185), 1, cv2.LINE_AA)

    for g in glyphs:
        x1, y1, x2, y2 = g.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (38, 160, 82), 1)
        cv2.putText(annotated, str(g.glyph_id + 1), (x1 + 2, max(12, y1 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (43, 56, 26), 1)
    return annotated


def _serialize_glyphs(glyphs: List[SegmentedGlyph]) -> List[Dict[str, Any]]:
    """Convert glyph objects into JSON-serializable dicts."""
    return [
        {
            "id": g.glyph_id,
            "bbox": [int(v) for v in g.bbox],
            "image": encode_image(g.to_fixed_size((64, 64))),
            "upper_modifier": g.has_upper_modifier,
            "lower_modifier": g.has_lower_modifier,
            "width": int(g.bbox[2] - g.bbox[0]),
            "height": int(g.bbox[3] - g.bbox[1]),
        }
        for g in glyphs
    ]


def _serialize_line(clean: np.ndarray, line_region: Tuple[int, int], line_id: int,
                     glyphs: List[SegmentedGlyph]) -> Dict[str, Any]:
    """Build the JSON payload for a single line."""
    annotated = _annotate_line(clean, line_region, glyphs)
    shirorekha = glyphs[0].metadata.get("shirorekha_y", 0) if glyphs else 0
    baseline = glyphs[0].metadata.get("baseline_y", 0) if glyphs else 0
    return {
        "id": int(line_id),
        "span": [int(line_region[0]), int(line_region[1])],
        "annotated": encode_image(annotated),
        "glyphs": _serialize_glyphs(glyphs),
        "shirorekha_y": int(shirorekha),
        "baseline_y": int(baseline),
    }


def line_debug_images(clean: np.ndarray, x_stretch: int, y_stretch: int, pad: int) -> tuple[dict[str, str], list[tuple[int, int]]]:
    """Reproduces the six visual stages from segmentation_visualizer.py."""
    binary_inv = cv2.adaptiveThreshold(
        clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    smeared = cv2.GaussianBlur(binary_inv, (x_stretch | 1, y_stretch | 1), 0)
    _, blobs = cv2.threshold(smeared, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_view = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    regions: list[tuple[int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > clean.shape[1] * 0.1 and h > 10:
            regions.append((y, y + h))
            cv2.rectangle(contour_view, (x, y), (x + w, y + h), (25, 133, 79), 2)
    regions.sort(key=lambda region: region[0])

    mask = np.zeros_like(clean)
    for start, end in regions:
        mask[max(0, start - pad):min(clean.shape[0], end + pad), :] = 255
    masked = cv2.bitwise_and(clean, clean, mask=mask)

    gap = 22
    assembled_height = max(clean.shape[0], sum(end - start + gap * 2 for start, end in regions))
    assembled = np.full((assembled_height, clean.shape[1], 3), 31, dtype=np.uint8)
    offset = gap
    for index, (start, end) in enumerate(regions):
        crop = cv2.cvtColor(clean[start:end, :], cv2.COLOR_GRAY2BGR)
        if offset + crop.shape[0] > assembled.shape[0]:
            break
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (32, 167, 113), 2)
        cv2.putText(assembled, f"LINE {index + 1}", (10, max(14, offset - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 230, 205), 1)
        assembled[offset:offset + crop.shape[0], :] = crop
        offset += crop.shape[0] + gap * 2

    return {
        "deskewed": encode_image(clean),
        "anisotropic_smear": encode_image(smeared),
        "blob_mask": encode_image(blobs),
        "contours": encode_image(contour_view),
        "iterative_mask": encode_image(masked),
        "extracted_lines": encode_image(assembled),
    }, regions


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Any:
    return {"status": "ok", "service": "Modi ShatVidhi pipeline"}


@app.post("/api/pipeline/process")
def process_document() -> Any:
    """Run the full preprocessing + line + character segmentation pipeline.
    
    Returns all visual intermediates plus a session_id for interactive
    merge/split/reset operations.
    """
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload an image of a Modi document."}), 400
    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if original is None:
        return jsonify({"error": "The uploaded file is not a readable image."}), 400

    block_size = int_value("block_size", 61, 3, 151) | 1
    adaptive_c = int_value("adaptive_c", 21, 0, 50)
    x_stretch = int_value("x_stretch", 41, 5, 301) | 1
    y_stretch = int_value("y_stretch", 5, 1, 51) | 1
    padding = int_value("padding", 5, 0, 30)
    cluster_multiplier = float(request.form.get("cluster_multiplier", 1.6))
    auto_tune = request.form.get("auto_tune", "true").lower() == "true"

    try:
        clean, intermediates = preprocess_image_array(
            original, PreprocessConfig(adaptive_block_size=block_size, adaptive_c=adaptive_c), return_intermediates=True
        )
        fft = None
        if auto_tune:
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                    temp_path = temp.name
                    cv2.imwrite(temp_path, clean)
                fft = FFTAnalyzer.calculate_optimal_parameters(temp_path)
                if fft:
                    x_stretch = int(fft.get("x_stretch", x_stretch)) | 1
                    y_stretch = int(fft.get("y_stretch", y_stretch)) | 1
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        line_stages, regions = line_debug_images(clean, x_stretch, y_stretch, padding)

        # Character segmentation
        segmenter = CharacterSegmenter(CharSegConfig(cluster_width_multiplier=max(1.1, min(cluster_multiplier, 2.5))))
        all_glyphs: List[List[SegmentedGlyph]] = []
        all_glyphs_backup: List[List[SegmentedGlyph]] = []
        lines_payload: List[Dict[str, Any]] = []

        for line_id, (start, end) in enumerate(regions):
            crop = clean[max(0, start):min(clean.shape[0], end), :]
            glyphs, _debug = segmenter.segment_line(crop, line_id)
            all_glyphs.append(list(glyphs))
            all_glyphs_backup.append(list(glyphs))
            lines_payload.append(_serialize_line(clean, (start, end), line_id, glyphs))

        glyph_count = sum(len(line["glyphs"]) for line in lines_payload)

        # Store session state for interactive merge/split and live parameter updates
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "clean": clean,
            "regions": regions,
            "glyphs": all_glyphs,
            "glyphs_backup": all_glyphs_backup,
            "segmenter": segmenter,
            "block_size": block_size,
            "adaptive_c": adaptive_c,
        }

        return jsonify({
            "session_id": session_id,
            "source": {"image": encode_image(original), "width": int(original.shape[1]), "height": int(original.shape[0])},
            "preprocessing": {
                "grayscale": encode_image(intermediates["grayscale"]), "denoised": encode_image(intermediates["denoised"]),
                "binary": encode_image(intermediates["binary"]), "cleaned": encode_image(intermediates["cleaned"]),
                "deskewed": encode_image(intermediates["deskewed"]),
            },
            "line_segmentation": line_stages,
            "characters": lines_payload,
            "summary": {
                "lines": len(lines_payload), "glyphs": glyph_count,
                "parameters": {
                    "block_size": block_size, "adaptive_c": adaptive_c,
                    "x_stretch": x_stretch, "y_stretch": y_stretch,
                    "auto_tune": auto_tune, "fft": fft,
                },
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/pipeline/merge")
def merge_glyphs() -> Any:
    """Merge selected adjacent glyphs on a line.
    
    Expects JSON: { session_id, line_id, glyph_indices: [int, ...] }
    Returns the updated line payload (annotated image + new glyph cards).
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    session_id = data.get("session_id", "")
    line_id = int(data.get("line_id", -1))
    indices = data.get("glyph_indices", [])

    if not indices or len(indices) < 2:
        return jsonify({"error": "Select at least 2 adjacent glyphs to merge."}), 400

    try:
        session = _get_session(session_id)
        clean = session["clean"]
        regions = session["regions"]
        segmenter: CharacterSegmenter = session["segmenter"]
        all_glyphs = session["glyphs"]

        if line_id < 0 or line_id >= len(regions):
            return jsonify({"error": f"Invalid line_id: {line_id}"}), 400

        sy, ey = regions[line_id]
        sy_c, ey_c = max(0, sy), min(clean.shape[0], ey)
        line_crop = clean[sy_c:ey_c, :]

        updated = segmenter.merge_glyph_spans(line_crop, all_glyphs[line_id], indices, line_id=line_id)
        all_glyphs[line_id] = updated

        line_payload = _serialize_line(clean, (sy, ey), line_id, updated)
        return jsonify({"line": line_payload, "total_glyphs": len(updated)})

    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/pipeline/split")
def split_glyph() -> Any:
    """Split an under-segmented glyph into two.
    
    Expects JSON: { session_id, line_id, glyph_idx, split_ratio? }
    Returns the updated line payload.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    session_id = data.get("session_id", "")
    line_id = int(data.get("line_id", -1))
    glyph_idx = int(data.get("glyph_idx", -1))
    split_ratio = float(data.get("split_ratio", 0.50))

    try:
        session = _get_session(session_id)
        clean = session["clean"]
        regions = session["regions"]
        segmenter: CharacterSegmenter = session["segmenter"]
        all_glyphs = session["glyphs"]

        if line_id < 0 or line_id >= len(regions):
            return jsonify({"error": f"Invalid line_id: {line_id}"}), 400

        sy, ey = regions[line_id]
        sy_c, ey_c = max(0, sy), min(clean.shape[0], ey)
        line_crop = clean[sy_c:ey_c, :]

        updated = segmenter.split_glyph_span(line_crop, all_glyphs[line_id], glyph_idx, split_ratio, line_id=line_id)
        all_glyphs[line_id] = updated

        line_payload = _serialize_line(clean, (sy, ey), line_id, updated)
        return jsonify({"line": line_payload, "total_glyphs": len(updated)})

    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/pipeline/reset-line")
def reset_line() -> Any:
    """Reset a line's glyphs to the initial algorithmic segmentation.
    
    Expects JSON: { session_id, line_id }
    Returns the restored line payload.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    session_id = data.get("session_id", "")
    line_id = int(data.get("line_id", -1))

    try:
        session = _get_session(session_id)
        clean = session["clean"]
        regions = session["regions"]
        all_glyphs = session["glyphs"]
        backup = session["glyphs_backup"]

        if line_id < 0 or line_id >= len(regions):
            return jsonify({"error": f"Invalid line_id: {line_id}"}), 400

        # Restore from backup
        all_glyphs[line_id] = list(backup[line_id])
        sy, ey = regions[line_id]

        line_payload = _serialize_line(clean, (sy, ey), line_id, all_glyphs[line_id])
        return jsonify({"line": line_payload, "total_glyphs": len(all_glyphs[line_id])})

    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/pipeline/update-lines")
def update_lines() -> Any:
    """Fast in-memory line and character re-segmentation when line controls change.
    
    Operates directly on the preprocessed clean image stored in the session,
    providing real-time feedback without re-uploading or re-running full preprocessing.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    session_id = data.get("session_id", "")
    try:
        session = _get_session(session_id)
        clean = session["clean"]

        x_stretch = int(data.get("x_stretch", 41)) | 1
        y_stretch = int(data.get("y_stretch", 5)) | 1
        padding = int(data.get("padding", 5))
        cluster_multiplier = float(data.get("cluster_multiplier", 1.6))
        auto_tune = bool(data.get("auto_tune", False))

        fft = None
        if auto_tune:
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                    temp_path = temp.name
                    cv2.imwrite(temp_path, clean)
                fft = FFTAnalyzer.calculate_optimal_parameters(temp_path)
                if fft:
                    x_stretch = int(fft.get("x_stretch", x_stretch)) | 1
                    y_stretch = int(fft.get("y_stretch", y_stretch)) | 1
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        line_stages, regions = line_debug_images(clean, x_stretch, y_stretch, padding)

        segmenter = CharacterSegmenter(CharSegConfig(cluster_width_multiplier=max(1.1, min(cluster_multiplier, 2.5))))
        all_glyphs: List[List[SegmentedGlyph]] = []
        all_glyphs_backup: List[List[SegmentedGlyph]] = []
        lines_payload: List[Dict[str, Any]] = []

        for line_id, (start, end) in enumerate(regions):
            crop = clean[max(0, start):min(clean.shape[0], end), :]
            glyphs, _debug = segmenter.segment_line(crop, line_id)
            all_glyphs.append(list(glyphs))
            all_glyphs_backup.append(list(glyphs))
            lines_payload.append(_serialize_line(clean, (start, end), line_id, glyphs))

        glyph_count = sum(len(line["glyphs"]) for line in lines_payload)

        # Update session state with the new segmentation
        session["regions"] = regions
        session["glyphs"] = all_glyphs
        session["glyphs_backup"] = all_glyphs_backup
        session["segmenter"] = segmenter

        return jsonify({
            "line_segmentation": line_stages,
            "characters": lines_payload,
            "summary": {
                "lines": len(lines_payload),
                "glyphs": glyph_count,
                "parameters": {
                    "block_size": session.get("block_size", 61),
                    "adaptive_c": session.get("adaptive_c", 21),
                    "x_stretch": x_stretch,
                    "y_stretch": y_stretch,
                    "auto_tune": auto_tune,
                    "fft": fft,
                },
            },
        })
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="127.0.0.1", port=port, debug=True)
