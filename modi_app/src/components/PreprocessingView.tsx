"use client";

import type { PreprocessingStages } from "../types/pipeline";

const imageLabels: Record<string, [string, string]> = {
  grayscale: [
    "Grayscale",
    "Normalises colour information to a single luminance channel.",
  ],
  denoised: [
    "Bilateral denoise",
    "Smooths paper texture while protecting stroke edges.",
  ],
  binary: [
    "Adaptive binary",
    "Separates ink from uneven historical-paper illumination.",
  ],
  cleaned: [
    "Speckle filter",
    "Removes small isolated connected components.",
  ],
  deskewed: [
    "Deskewed output",
    "Horizontal-projection variance estimates and corrects page tilt.",
  ],
};

interface PreprocessingViewProps {
  stages: PreprocessingStages;
  onOpenImage: (src: string, title: string) => void;
}

export function PreprocessingView({
  stages,
  onOpenImage,
}: PreprocessingViewProps) {
  const entries = Object.entries(stages) as [
    keyof PreprocessingStages,
    string,
  ][];

  return (
    <section className="stage-pane" id="preprocessing">
      <div className="view-intro">
        <div>
          <p className="eyebrow">
            Desktop visualizer, translated for web
          </p>
          <h3>Preprocessing visualizer</h3>
          <p>
            Each frame below is emitted by preprocessing/preprocess.py and
            mirrors the desktop preprocessing visualizer.
          </p>
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
