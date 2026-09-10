"use client";

import { ChangeEvent, FormEvent } from "react";
import type { PipelineParams } from "../types/pipeline";

interface ControlPanelProps {
  file: File | null;
  preview: string;
  params: PipelineParams;
  busy: boolean;
  lineBusy?: boolean;
  isProcessed?: boolean;
  error: string;
  onSelectFile: (e: ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: () => void;
  onParamsChange: (params: PipelineParams) => void;
  onRun: (e: FormEvent) => void;
}

function Range({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="range">
      <span>
        {label}
        <b>{value}</b>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function ControlPanel({
  file,
  preview,
  params,
  busy,
  lineBusy = false,
  isProcessed = false,
  error,
  onSelectFile,
  onRemoveFile,
  onParamsChange,
  onRun,
}: ControlPanelProps) {
  const set = (patch: Partial<PipelineParams>) =>
    onParamsChange({ ...params, ...patch });

  return (
    <aside className="control-panel">
      <div className="panel-title">
        <span>01</span>
        <div>
          <b>Document input</b>
          <small>Configure &amp; run</small>
        </div>
      </div>

      <form onSubmit={onRun}>
        {/* ── Dropzone ── */}
        <label className={`dropzone ${preview ? "has-file" : ""}`}>
          <input
            type="file"
            accept="image/png,image/jpeg,image/tiff,image/bmp"
            onChange={onSelectFile}
          />
          {preview ? (
            <img src={preview} alt="Selected manuscript preview" />
          ) : (
            <>
              <span>↥</span>
              <b>Choose manuscript image</b>
              <small>PNG, JPG, TIFF or BMP · 15 MB max</small>
            </>
          )}
        </label>

        {file && (
          <p className="file-name">
            {file.name}
            <button type="button" onClick={onRemoveFile}>
              Remove
            </button>
          </p>
        )}

        {/* ── Preprocessing ── */}
        <fieldset>
          <legend>Preprocessing</legend>
          <Range
            label="Block size"
            value={params.block_size}
            min={3}
            max={151}
            step={2}
            onChange={(v) => set({ block_size: v | 1 })}
          />
          <Range
            label="Adaptive C"
            value={params.adaptive_c}
            min={0}
            max={50}
            step={1}
            onChange={(v) => set({ adaptive_c: v })}
          />
        </fieldset>

        {/* ── Line localisation ── */}
        <fieldset>
          <legend className="fieldset-legend">
            <span>Line localisation</span>
            {lineBusy && <span className="live-pill active">⚡ Updating lines…</span>}
            {!lineBusy && isProcessed && <span className="live-pill idle">⚡ Real-time</span>}
          </legend>
          <label className="toggle">
            <input
              type="checkbox"
              checked={params.auto_tune}
              onChange={(e) => set({ auto_tune: e.target.checked })}
            />
            <span /> ACF auto-tune
          </label>
          <Range
            label="X stretch"
            value={params.x_stretch}
            min={5}
            max={301}
            step={2}
            onChange={(v) => set({ x_stretch: v | 1, auto_tune: false })}
          />
          <Range
            label="Y stretch"
            value={params.y_stretch}
            min={1}
            max={51}
            step={2}
            onChange={(v) => set({ y_stretch: v | 1, auto_tune: false })}
          />
          <Range
            label="Padding"
            value={params.padding ?? 5}
            min={0}
            max={30}
            step={1}
            onChange={(v) => set({ padding: v })}
          />
        </fieldset>

        {/* ── Character segmentation ── */}
        <fieldset>
          <legend>Character segmentation</legend>
          <Range
            label="Cluster split"
            value={params.cluster_multiplier}
            min={1.1}
            max={2.5}
            step={0.05}
            onChange={(v) => set({ cluster_multiplier: v })}
          />
        </fieldset>

        <button className="run-button" disabled={busy}>
          {busy ? (
            <>
              <span className="spinner" />
              Processing manuscript…
            </>
          ) : (
            <>
              Run Modi ShatVidhi <span>→</span>
            </>
          )}
        </button>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>

      <div className="run-notes">
        <b>What runs locally</b>
        <p>
          OpenCV preprocessing, FFT/ACF tuning, contour-based line
          segmentation, and the Modi-aware glyph segmenter.
        </p>
      </div>
    </aside>
  );
}
