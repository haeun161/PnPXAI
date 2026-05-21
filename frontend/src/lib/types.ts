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
}

export interface PredictionItem {
  class_name: string;
  probability: number;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "partial" | "completed" | "failed";
  task: TaskType;
  model_name: string;
  explainer_names: string[];
  ranking_metric: string;
  predictions: PredictionItem[] | null;
  original_data_url: string | null;
  results: ExplainerResult[];
  error_message: string | null;
}
