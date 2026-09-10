"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PipelineParams, PipelineResult, TabKey } from "../types/pipeline";
import { usePipeline } from "../hooks/usePipeline";
import { ControlPanel } from "./ControlPanel";
import { PreprocessingView } from "./PreprocessingView";
import { LineSegmentationView } from "./LineSegmentationView";
import { GlyphView } from "./GlyphView";
import { ImageModal } from "./ImageModal";

/* ── Pipeline stage descriptions ── */
const stageInfo: [string, string, string][] = [
  ["01", "Acquire", "Upload a colour or grayscale scan. The server decodes it with OpenCV without altering your original file."],
  ["02", "Prepare", "Bilateral denoising, adaptive thresholding, component filtering, and HPP deskewing create a clean manuscript surface."],
  ["03", "Locate lines", "Autocorrelation estimates spacing; an anisotropic Gaussian merge finds text-line regions across the page."],
  ["04", "Segment glyphs", "Zonal analysis protects matras, then ligature, cluster, and kana rules form complete Modi akshara candidates."],
  ["05", "Recognise", "A recognition/transliteration model can consume the normalized 64×64 glyphs (Phase 4+)."],
];

const tabs: [TabKey, string][] = [
  ["overview", "Overview"],
  ["preprocessing", "Preprocessing"],
  ["lines", "Lines"],
  ["glyphs", "Glyphs"],
  ["recognition", "Transliteration"],
];

export function PipelineWorkspace() {
  /* ── Local UI state ── */
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [tab, setTab] = useState<TabKey>("overview");
  const [activeLine, setActiveLine] = useState(0);
  const [zoom, setZoom] = useState<{ src: string; title: string } | null>(null);
  const [params, setParams] = useState<PipelineParams>({
    block_size: 61,
    adaptive_c: 21,
    x_stretch: 41,
    y_stretch: 5,
    padding: 5,
    cluster_multiplier: 1.6,
    auto_tune: true,
  });

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  /* ── Pipeline hook ── */
  const {
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
  } = usePipeline();

  // Keep parameters in sync when document is first processed (e.g. ACF detected parameters)
  useEffect(() => {
    if (result?.summary?.parameters) {
      setParams((prev) => ({
        ...prev,
        x_stretch: result.summary.parameters.x_stretch,
        y_stretch: result.summary.parameters.y_stretch,
        auto_tune: result.summary.parameters.auto_tune,
      }));
    }
  }, [result?.session_id]);

  const handleParamsChange = useCallback(
    (newParams: PipelineParams) => {
      setParams(newParams);

      // Real-time line update: if manuscript already loaded, tweak line extraction immediately
      if (result) {
        const lineParamsChanged =
          newParams.x_stretch !== params.x_stretch ||
          newParams.y_stretch !== params.y_stretch ||
          newParams.padding !== params.padding ||
          newParams.cluster_multiplier !== params.cluster_multiplier ||
          newParams.auto_tune !== params.auto_tune;

        if (lineParamsChanged) {
          if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
          debounceTimerRef.current = setTimeout(() => {
            updateLineParameters(newParams);
          }, 200);
        }
      }
    },
    [result, params, updateLineParameters]
  );

  const dataReady = !!result;

  /* ── Derived stats ── */
  const stats = useMemo(
    () =>
      result
        ? [
            ["Lines located", result.summary.lines],
            ["Glyph candidates", result.summary.glyphs],
            ["Image", `${result.source.width} × ${result.source.height}`],
          ]
        : [
            ["Phase", "1–4"],
            ["Output", "Visual"],
            ["Mode", "Local"],
          ],
    [result]
  );

  /* ── Handlers ── */
  function selectFile(e: ChangeEvent<HTMLInputElement>) {
    const next = e.target.files?.[0] || null;
    setFile(next);
    clearResult();
    setTab("overview");
    setPreview(next ? URL.createObjectURL(next) : "");
  }

  function removeFile() {
    setFile(null);
    setPreview("");
    clearResult();
  }

  async function runPipeline(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    await processDocument(file, params);
    setTab("preprocessing");
    setActiveLine(0);
  }

  const openImage = (src: string, title: string) => setZoom({ src, title });

  return (
    <main className="pipeline-app">
      {/* ── Header ── */}
      <header className="app-header">
        <a className="brand" href="#workspace">
          <span className="seal">श्री</span>
          <span>
            <b>Modi ShatVidhi</b>
            <em>Transliteration workspace</em>
          </span>
        </a>
        <div className="header-status">
          <span className="pulse" />
          Python pipeline <small>OpenCV · local processing</small>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="app-hero">
        <div>
          <p className="eyebrow">
            A web interface for the manuscript-analysis pipeline
          </p>
          <h1>
            From <i>Modi manuscript</i>
            <br />
            to meaningful reading.
          </h1>
          <p>
            Upload an archaic Modi document and inspect every processing
            decision — from paper cleanup and line localisation to complete
            glyph candidates for transliteration.
          </p>
        </div>
        <aside>
          <span className="hero-quote">
            &ldquo;प्रतिपच्चंद्रलेखेव वर्धिष्णुर्विश्ववन्दिता&rdquo;
          </span>
          <p>
            Built around the existing Modi ShatVidhi desktop algorithms, now
            made visible on the web.
          </p>
        </aside>
      </section>

      {/* ── Workflow strip ── */}
      <section className="workflow-strip" aria-label="Pipeline steps">
        {stageInfo.map(([num, title], idx) => (
          <a
            href={
              idx < 4
                ? `#${["upload", "preprocessing", "lines", "glyphs"][idx]}`
                : "#recognition"
            }
            key={title}
            className={dataReady && idx < 4 ? "complete" : ""}
          >
            <span>{num}</span>
            <b>{title}</b>
            <i>→</i>
          </a>
        ))}
      </section>

      {/* ── Main workspace ── */}
      <div className="workspace" id="workspace">
        <ControlPanel
          file={file}
          preview={preview}
          params={params}
          busy={busy}
          lineBusy={lineBusy}
          isProcessed={!!result}
          error={error}
          onSelectFile={selectFile}
          onRemoveFile={removeFile}
          onParamsChange={handleParamsChange}
          onRun={runPipeline}
        />

        <section className="result-area">
          {/* Top summary */}
          <div className="result-top">
            <div>
              <p className="eyebrow">
                {result ? "Analysis complete" : "Ready for a document"}
              </p>
              <h2>
                {result
                  ? `${result.summary.lines} lines · ${result.summary.glyphs} glyph candidates`
                  : "Your analysis workspace"}
              </h2>
            </div>
            <div className="metrics">
              {stats.map(([label, value]) => (
                <div key={label as string}>
                  <b>{String(value)}</b>
                  <small>{String(label)}</small>
                </div>
              ))}
            </div>
          </div>

          {!result ? (
            <EmptyState />
          ) : (
            <>
              {/* Tabs */}
              <nav className="tabs" aria-label="Pipeline output views">
                {tabs.map(([key, label]) => (
                  <button
                    key={key}
                    className={tab === key ? "selected" : ""}
                    onClick={() => setTab(key)}
                  >
                    {label}
                  </button>
                ))}
              </nav>

              {/* Tab content */}
              {tab === "overview" && (
                <Overview result={result} onOpenImage={openImage} />
              )}
              {tab === "preprocessing" && (
                <PreprocessingView
                  stages={result.preprocessing}
                  onOpenImage={openImage}
                />
              )}
              {tab === "lines" && (
                <LineSegmentationView
                  stages={result.line_segmentation}
                  parameters={result.summary.parameters}
                  onOpenImage={openImage}
                />
              )}
              {tab === "glyphs" && (
                <GlyphView
                  lines={result.characters}
                  activeLine={
                    result.characters.length > 0
                      ? Math.min(activeLine, result.characters.length - 1)
                      : 0
                  }
                  setActiveLine={setActiveLine}
                  actionBusy={actionBusy || lineBusy}
                  onMerge={mergeGlyphs}
                  onSplit={splitGlyph}
                  onReset={resetLine}
                  onOpenImage={openImage}
                />
              )}
              {tab === "recognition" && <Recognition result={result} />}
            </>
          )}
        </section>
      </div>

      {/* ── Image lightbox ── */}
      {zoom && (
        <ImageModal
          src={zoom.src}
          title={zoom.title}
          onClose={() => setZoom(null)}
        />
      )}
    </main>
  );
}

/* ── Sub-components ── */

function EmptyState() {
  return (
    <div className="empty-workspace">
      <div className="empty-seal">श्री</div>
      <h3>The document journey begins here.</h3>
      <p>
        Upload a scan from the control panel. Its real intermediate images will
        replace this panel after the Python pipeline completes.
      </p>
      <div>
        {stageInfo.slice(0, 4).map(([number, title, description]) => (
          <article key={number}>
            <span>{number}</span>
            <div>
              <b>{title}</b>
              <p>{description}</p>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function Overview({
  result,
  onOpenImage,
}: {
  result: PipelineResult;
  onOpenImage: (src: string, title: string) => void;
}) {
  return (
    <section className="overview-pane">
      <div className="source-card">
        <div>
          <p className="eyebrow">Source document</p>
          <h3>Processed manuscript</h3>
          <p>
            {result.source.width} × {result.source.height} pixels · Analysis is
            kept in-memory for this request.
          </p>
          <button
            onClick={() =>
              onOpenImage(result.source.image, "Original uploaded manuscript")
            }
          >
            View full image ↗
          </button>
        </div>
        <img src={result.source.image} alt="Original uploaded Modi manuscript" />
      </div>

      <div className="handoff-cards">
        <article>
          <span>02</span>
          <h3>Image conditioning</h3>
          <p>
            Five visible outputs document the noise-reduction, adaptive
            threshold, and deskew decisions.
          </p>
          <a href="#preprocessing">Inspect preprocessing →</a>
        </article>
        <article>
          <span>03</span>
          <h3>Text-line structure</h3>
          <p>
            Gaussian blobs and contours make the line-localisation decision
            auditable.
          </p>
          <a href="#lines">Inspect line extraction →</a>
        </article>
        <article>
          <span>04</span>
          <h3>Akshara candidates</h3>
          <p>
            Every line exposes normalized 64×64 glyph crops for interactive
            merge/split refinement.
          </p>
          <a href="#glyphs">Inspect &amp; refine glyph candidates →</a>
        </article>
      </div>
    </section>
  );
}

function Recognition({ result }: { result: PipelineResult }) {
  return (
    <section className="recognition" id="recognition">
      <div className="recognition-mark">05</div>
      <div>
        <p className="eyebrow">The model boundary is explicit</p>
        <h3>Ready for transliteration recognition.</h3>
        <p>
          The supplied project implements visual preprocessing and glyph
          segmentation, producing{" "}
          <b>{result.summary.glyphs} normalized glyph candidates</b>. It does
          not include trained OCR, VLM, or Modi-to-Devanagari weights, so this
          UI deliberately does not invent a transliteration.
        </p>
        <div className="recognition-flow">
          <span>64 × 64 glyph tensors</span>
          <i>→</i>
          <span>Recognition model</span>
          <i>→</i>
          <span>Devanagari / Roman output</span>
        </div>
        <div className="model-note">
          <b>Next integration point</b>
          <p>
            Connect the Phase 4 model endpoint here. The backend can forward
            each glyph crop, retain line order, and render confidence scores
            alongside editable transliteration text.
          </p>
        </div>
      </div>
    </section>
  );
}
