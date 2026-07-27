export type TaskType = "image" | "text" | "timeseries";

export interface TaskInfo {
  name: TaskType;
  display_name: string;
  description: string;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  architecture: string;
  description: string;
  task: string;
}

export interface ExplainerInfo {
  name: string;
  display_name: string;
  estimated_compute_time_seconds: number;
  compatible: boolean;
  incompatibility_reason: string | null;
}

export interface TokenAttribution {
  token: string;
  score: number;
}

export interface ExplainerResult {
  explainer_name: string;
  display_name: string;
  status: "completed" | "running" | "pending" | "not_supported" | "failed";
  rank: number | null;
  visualization_url: string | null;
  mu_fidelity: number | null;
  abpc: number | null;
  sensitivity: number | null;
  complexity: number | null;
  not_supported_reason: string | null;
  error_message: string | null;
  token_attributions: TokenAttribution[] | null;
  current_step: string | null;
}

// A selectable sample data file. The server already filters this list to what the
// chosen model was trained on, so the client never sees the model→dataset mapping.
export interface SampleFile {
  name: string;
  path: string;
  compatible?: boolean;
  reason?: string;
  channels?: number;
  col_names?: string[];
}

export interface PredictionItem {
  class_name: string;
  probability: number;
}

// Back-test for forecasting models: `context` is the input window, `predicted` the
// horizon the model produced from it, and `actual` the observed values over that same
// horizon (null when the forecast runs past the end of the uploaded data).
// All series are [timestep][channel].
export interface ForecastInfo {
  col_names: string[];
  context: number[][];
  predicted: number[][];
  actual: number[][] | null;
  context_labels: string[] | null;
  horizon_labels: string[] | null;
  attributed_channel: number;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "partial" | "completed" | "failed" | "cancelled";
  task: TaskType;
  model_name: string;
  explainer_names: string[];
  ranking_metric: string;
  predictions: PredictionItem[] | null;
  forecast: ForecastInfo | null;
  original_data_url: string | null;
  results: ExplainerResult[];
  error_message: string | null;
}
