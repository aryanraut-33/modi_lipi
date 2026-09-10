"use client";

import { useCallback, useState } from "react";
import type { Glyph, Line } from "../types/pipeline";

interface GlyphViewProps {
  lines: Line[];
  activeLine: number;
  setActiveLine: (id: number) => void;
  actionBusy: boolean;
  onMerge: (lineId: number, indices: number[]) => Promise<void>;
  onSplit: (lineId: number, glyphIdx: number) => Promise<void>;
  onReset: (lineId: number) => Promise<void>;
  onOpenImage: (src: string, title: string) => void;
}

export function GlyphView({
  lines,
  activeLine,
  setActiveLine,
  actionBusy,
  onMerge,
  onSplit,
  onReset,
  onOpenImage,
}: GlyphViewProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [toast, setToast] = useState("");

  const current = lines[activeLine];

  const clearSelection = useCallback(() => setSelected(new Set()), []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }, []);

  const toggleGlyph = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleMergeChecked = async () => {
    const indices = Array.from(selected).sort((a, b) => a - b);
    if (indices.length < 2) {
      showToast("Select at least 2 adjacent glyphs to merge.");
      return;
    }
    await onMerge(activeLine, indices);
    clearSelection();
    showToast(`Merged ${indices.length} glyphs into one character.`);
  };

  const handleMergeLeft = async (idx: number) => {
    if (idx <= 0) return;
    await onMerge(activeLine, [idx - 1, idx]);
    clearSelection();
    showToast(`Merged glyph #${idx} with #${idx + 1}.`);
  };

  const handleMergeRight = async (idx: number) => {
    if (!current || idx >= current.glyphs.length - 1) return;
    await onMerge(activeLine, [idx, idx + 1]);
    clearSelection();
    showToast(`Merged glyph #${idx + 1} with #${idx + 2}.`);
  };

  const handleSplit = async (idx: number) => {
    await onSplit(activeLine, idx);
    clearSelection();
    showToast(`Split glyph #${idx + 1} into two characters.`);
  };

  const handleReset = async () => {
    await onReset(activeLine);
    clearSelection();
    showToast(`Line ${activeLine + 1} restored to initial segmentation.`);
  };

  const handleLineChange = (id: number) => {
    setActiveLine(id);
    clearSelection();
  };

  return (
    <section className="glyph-pane" id="glyphs">
      {/* ── Header ── */}
      <div className="view-intro">
        <div>
          <p className="eyebrow">Modi-aware character segmentation</p>
          <h3>Interactive glyph refinement</h3>
          <p>
            Green boxes mark extracted characters. Select adjacent glyphs to
            merge over-segmented fragments, or split wide clusters into
            individual letters.
          </p>
        </div>
      </div>

      {/* ── Line picker ── */}
      <div className="line-picker" role="tablist" aria-label="Detected lines">
        {lines.map((line) => (
          <button
            key={line.id}
            role="tab"
            aria-selected={line.id === activeLine}
            className={line.id === activeLine ? "selected" : ""}
            onClick={() => handleLineChange(line.id)}
          >
            Line {line.id + 1}
            <small>{line.glyphs.length} glyphs</small>
          </button>
        ))}
      </div>

      {current ? (
        <>
          {/* ── Toolbar ── */}
          <div className="glyph-toolbar">
            <button
              className="toolbar-btn merge-btn"
              disabled={actionBusy || selected.size < 2}
              onClick={handleMergeChecked}
            >
              🔗 Merge Checked ({selected.size})
            </button>
            <button
              className="toolbar-btn reset-btn"
              disabled={actionBusy}
              onClick={handleReset}
            >
              ↺ Reset Line
            </button>
            <span className="toolbar-info">
              Line {activeLine + 1} · {current.glyphs.length} glyphs
              {selected.size > 0 && ` · ${selected.size} selected`}
            </span>
            {toast && <span className="toolbar-toast">{toast}</span>}
          </div>

          {/* ── Annotated line banner ── */}
          <button
            className="annotated-line"
            onClick={() =>
              onOpenImage(
                current.annotated,
                `Line ${current.id + 1}: segmentation overlay`
              )
            }
          >
            <img
              src={current.annotated}
              alt={`Annotated segmentation of line ${current.id + 1}`}
            />
            <span>View segmentation overlay ↗</span>
          </button>

          {/* ── Glyph grid with interactive cards ── */}
          <div className="glyph-grid interactive">
            {current.glyphs.map((glyph: Glyph, idx: number) => {
              const isSelected = selected.has(idx);
              return (
                <article
                  key={`${current.id}-${glyph.id}-${idx}`}
                  className={`glyph-card ${isSelected ? "selected" : ""}`}
                >
                  {/* Checkbox */}
                  <label className="glyph-check">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleGlyph(idx)}
                    />
                    <span>#{glyph.id + 1}</span>
                  </label>

                  {/* 64x64 image */}
                  <img
                    src={glyph.image}
                    alt={`Glyph ${glyph.id + 1}`}
                    onClick={() => toggleGlyph(idx)}
                  />

                  {/* Info */}
                  <div className="glyph-info">
                    <b>{glyph.width}×{glyph.height}</b>
                    <small>
                      {glyph.upper_modifier ? "▲ Top" : ""}
                      {glyph.upper_modifier && glyph.lower_modifier ? " · " : ""}
                      {glyph.lower_modifier ? "▼ Bot" : ""}
                      {!glyph.upper_modifier && !glyph.lower_modifier
                        ? "Base only"
                        : ""}
                    </small>
                  </div>

                  {/* Action buttons */}
                  <div className="glyph-actions">
                    {idx > 0 && (
                      <button
                        className="act-btn"
                        disabled={actionBusy}
                        onClick={() => handleMergeLeft(idx)}
                        title="Merge with left neighbour"
                      >
                        ◂
                      </button>
                    )}
                    <button
                      className="act-btn split"
                      disabled={actionBusy}
                      onClick={() => handleSplit(idx)}
                      title="Split into two characters"
                    >
                      Split
                    </button>
                    {idx < current.glyphs.length - 1 && (
                      <button
                        className="act-btn"
                        disabled={actionBusy}
                        onClick={() => handleMergeRight(idx)}
                        title="Merge with right neighbour"
                      >
                        ▸
                      </button>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </>
      ) : (
        <p className="no-lines">
          No valid glyphs were found in this line. Adjust parameters and run
          again.
        </p>
      )}
    </section>
  );
}
