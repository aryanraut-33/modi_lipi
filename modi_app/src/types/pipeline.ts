/* ── Pipeline API response types ── */

export interface Glyph {
  id: number;
  bbox: [number, number, number, number];
  image: string;           // data:image/png;base64,...
  upper_modifier: boolean;
  lower_modifier: boolean;
  width: number;
  height: number;
}

export interface Line {
  id: number;
  span: [number, number];
  annotated: string;       // data:image/png;base64,...
  glyphs: Glyph[];
  shirorekha_y: number;
  baseline_y: number;
}

export interface FFTInfo {
  spatial_period_px?: number;
  x_stretch?: number;
  y_stretch?: number;
}

export interface PipelineParameters {
  block_size: number;
  adaptive_c: number;
  x_stretch: number;
  y_stretch: number;
  auto_tune: boolean;
  fft?: FFTInfo | null;
}

export interface PipelineSummary {
  lines: number;
  glyphs: number;
  parameters: PipelineParameters;
}

export interface PreprocessingStages {
  grayscale: string;
  denoised: string;
  binary: string;
  cleaned: string;
  deskewed: string;
}

export interface LineSegmentationStages {
  deskewed: string;
  anisotropic_smear: string;
  blob_mask: string;
  contours: string;
  iterative_mask: string;
  extracted_lines: string;
}

export interface SourceInfo {
  image: string;
  width: number;
  height: number;
}

export interface PipelineResult {
  session_id: string;
  source: SourceInfo;
  preprocessing: PreprocessingStages;
  line_segmentation: LineSegmentationStages;
  characters: Line[];
  summary: PipelineSummary;
}

export interface PipelineParams {
  block_size: number;
  adaptive_c: number;
  x_stretch: number;
  y_stretch: number;
  padding?: number;
  cluster_multiplier: number;
  auto_tune: boolean;
}

export interface LineUpdateResponse {
  line: Line;
  total_glyphs: number;
}

export interface LinesBatchUpdateResponse {
  line_segmentation: LineSegmentationStages;
  characters: Line[];
  summary: PipelineSummary;
}

export type TabKey = "overview" | "preprocessing" | "lines" | "glyphs" | "recognition";
