"use client";

import { useCallback, useRef, useState } from "react";
import type {
  Line,
  LineUpdateResponse,
  PipelineParams,
  PipelineResult,
} from "../types/pipeline";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5005";

interface UsePipelineReturn {
  result: PipelineResult | null;
  busy: boolean;
  error: string;
  actionBusy: boolean;
  lineBusy: boolean;
  processDocument: (file: File, params: PipelineParams) => Promise<void>;
  updateLineParameters: (params: PipelineParams) => Promise<void>;
  mergeGlyphs: (lineId: number, indices: number[]) => Promise<void>;
  splitGlyph: (lineId: number, glyphIdx: number) => Promise<void>;
  resetLine: (lineId: number) => Promise<void>;
  clearResult: () => void;
  updateLine: (lineId: number, line: Line) => void;
}

export function usePipeline(): UsePipelineReturn {
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [lineBusy, setLineBusy] = useState(false);
  const [error, setError] = useState("");
  const sessionRef = useRef<string>("");
  const lineAbortRef = useRef<AbortController | null>(null);

  const processDocument = useCallback(
    async (file: File, params: PipelineParams) => {
      setBusy(true);
      setError("");
      try {
        const form = new FormData();
        form.append("image", file);
        Object.entries(params).forEach(([k, v]) => form.append(k, String(v)));

        const res = await fetch(`${API_BASE}/api/pipeline/process`, {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Pipeline processing failed.");

        sessionRef.current = data.session_id;
        setResult(data as PipelineResult);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to reach the Python pipeline. Start backend/app.py on port 5005."
        );
      } finally {
        setBusy(false);
      }
    },
    []
  );

  const updateLine = useCallback((lineId: number, line: Line) => {
    setResult((prev) => {
      if (!prev) return prev;
      const chars = [...prev.characters];
      chars[lineId] = line;
      const totalGlyphs = chars.reduce((s, l) => s + l.glyphs.length, 0);
      return {
        ...prev,
        characters: chars,
        summary: { ...prev.summary, glyphs: totalGlyphs },
      };
    });
  }, []);

  const updateLineParameters = useCallback(
    async (params: PipelineParams) => {
      if (!sessionRef.current) return;
      if (lineAbortRef.current) {
        lineAbortRef.current.abort();
      }
      const controller = new AbortController();
      lineAbortRef.current = controller;
      setLineBusy(true);
      setError("");

      try {
        const res = await fetch(`${API_BASE}/api/pipeline/update-lines`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            session_id: sessionRef.current,
            x_stretch: params.x_stretch,
            y_stretch: params.y_stretch,
            cluster_multiplier: params.cluster_multiplier,
            auto_tune: params.auto_tune,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to update line parameters.");

        setResult((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            line_segmentation: data.line_segmentation,
            characters: data.characters,
            summary: {
              ...prev.summary,
              lines: data.summary.lines,
              glyphs: data.summary.glyphs,
              parameters: {
                ...prev.summary.parameters,
                ...data.summary.parameters,
              },
            },
          };
        });
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to update line parameters.");
      } finally {
        setLineBusy(false);
      }
    },
    []
  );

  const mergeGlyphs = useCallback(
    async (lineId: number, indices: number[]) => {
      if (!sessionRef.current) return;
      setActionBusy(true);
      setError("");
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/merge`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionRef.current,
            line_id: lineId,
            glyph_indices: indices,
          }),
        });
        const data: LineUpdateResponse = await res.json();
        if (!res.ok) throw new Error((data as any).error || "Merge failed.");
        updateLine(lineId, data.line);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Merge operation failed.");
      } finally {
        setActionBusy(false);
      }
    },
    [updateLine]
  );

  const splitGlyph = useCallback(
    async (lineId: number, glyphIdx: number) => {
      if (!sessionRef.current) return;
      setActionBusy(true);
      setError("");
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/split`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionRef.current,
            line_id: lineId,
            glyph_idx: glyphIdx,
          }),
        });
        const data: LineUpdateResponse = await res.json();
        if (!res.ok) throw new Error((data as any).error || "Split failed.");
        updateLine(lineId, data.line);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Split operation failed.");
      } finally {
        setActionBusy(false);
      }
    },
    [updateLine]
  );

  const resetLine = useCallback(
    async (lineId: number) => {
      if (!sessionRef.current) return;
      setActionBusy(true);
      setError("");
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/reset-line`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionRef.current,
            line_id: lineId,
          }),
        });
        const data: LineUpdateResponse = await res.json();
        if (!res.ok) throw new Error((data as any).error || "Reset failed.");
        updateLine(lineId, data.line);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Reset operation failed.");
      } finally {
        setActionBusy(false);
      }
    },
    [updateLine]
  );

  const clearResult = useCallback(() => {
    setResult(null);
    setError("");
    sessionRef.current = "";
  }, []);

  return {
    result,
    busy,
    error,
    actionBusy,
    lineBusy,
    processDocument,
    updateLineParameters,
    mergeGlyphs,
    splitGlyph,
    resetLine,
    clearResult,
    updateLine,
  };
}
