"use client";

import type { LineSegmentationStages, PipelineParameters } from "../types/pipeline";

const imageLabels: Record<string, [string, string]> = {
  deskewed: [
    "Deskewed input",
    "Clean grayscale image fed into line detection.",
  ],
  anisotropic_smear: [
    "Anisotropic smear",
    "Horizontal blur connects nearby glyphs within a line.",
  ],
  blob_mask: [
    "Solid line blobs",
    "Thresholding turns merged ink into candidate line regions.",
  ],
  contours: [
    "Contours",
    "Long horizontal regions are retained as line candidates.",
  ],
  iterative_mask: [
    "Iterative mask",
    "The line regions mask out non-text portions of the page.",
  ],
  extracted_lines: [
    "Extracted lines",
    "Final cut-line preview used by character segmentation.",
  ],
};

interface LineSegmentationViewProps {
  stages: LineSegmentationStages;
  parameters: PipelineParameters;
  onOpenImage: (src: string, title: string) => void;
}

export function LineSegmentationView({
  stages,
  parameters,
  onOpenImage,
}: LineSegmentationViewProps) {
  const entries = Object.entries(stages) as [
    keyof LineSegmentationStages,
    string,
  ][];

  const desc = `The ACF pass selected σx ${parameters.x_stretch} and σy ${parameters.y_stretch}${
    parameters.fft?.spatial_period_px
      ? ` from an estimated ${parameters.fft.spatial_period_px}px line period`
      : ""
  }.`;

  return (
    <section className="stage-pane" id="lines">
      <div className="view-intro">
        <div>
          <p className="eyebrow">
            Desktop visualizer, translated for web
          </p>
          <h3>Line segmentation visualizer</h3>
          <p>{desc}</p>
        </div>
        <span>Click an image to inspect it at full resolution.</span>
      </div>

      <div className="image-grid">
        {entries.map(([key, src], idx) => {
          const [title, text] = imageLabels[key] || [key, ""];
          return (
            <button
              key={key}
              className="stage-image"
              onClick={() => onOpenImage(src, title)}
            >
              <span>{String(idx + 1).padStart(2, "0")}</span>
              <img src={src} alt={title} />
              <div>
                <b>{title}</b>
                <small>{text}</small>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
