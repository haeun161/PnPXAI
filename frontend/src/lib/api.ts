import { TaskInfo, TaskType, ModelInfo, ExplainerInfo, JobStatus, SampleFile } from "./types";

const BASE = "/api";

export async function getTasks(): Promise<TaskInfo[]> {
  const res = await fetch(`${BASE}/tasks`);
  if (!res.ok) throw new Error("Failed to fetch tasks");
  return res.json();
}

export async function getModels(task: TaskType): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/models?task=${task}`);
  if (!res.ok) throw new Error("Failed to fetch models");
  return res.json();
}

export async function getExplainers(task: TaskType, model?: string): Promise<ExplainerInfo[]> {
  let url = `${BASE}/explainers?task=${task}`;
  if (model) url += `&model=${model}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch explainers");
  return res.json();
}

export async function getRecommendedExplainers(
  task: TaskType,
  model: string,
): Promise<{ recommended: string[]; detected_architectures: string[] }> {
  const res = await fetch(`${BASE}/recommend?task=${task}&model=${encodeURIComponent(model)}`);
  if (!res.ok) throw new Error("Failed to fetch recommendations");
  return res.json();
}

export async function validateHfModel(task: TaskType, hfModelId: string): Promise<{ model_id: string; display_name: string }> {
  const res = await fetch(`${BASE}/validate-model?task=${task}&hf_model_id=${encodeURIComponent(hfModelId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Validation failed" }));
    throw new Error(err.detail || "Validation failed");
  }
  return res.json();
}

export async function uploadModel(task: TaskType, file: File): Promise<{ model_id: string; display_name: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload-model?task=${task}`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function getSamples(task: TaskType, model?: string): Promise<SampleFile[]> {
  const url = model ? `${BASE}/samples/${task}?model=${encodeURIComponent(model)}` : `${BASE}/samples/${task}`;
  const res = await fetch(url);
  if (!res.ok) return [];
  return res.json();
}

export async function submitExplainJob(
  task: TaskType,
  file: File | Blob,
  modelName: string,
  explainerNames: string[],
  rankingMetric: string = "average",
  dataName?: string,
  tsWindowIndex?: number,
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({
    task,
    model_name: modelName,
    explainer_names: explainerNames.join(","),
    ranking_metric: rankingMetric,
  });
  // Lets the server pick the checkpoint trained on this dataset; absent for uploads.
  if (dataName) params.set("data_name", dataName);
  // Re-anchors the time-series backtest so this chained window's context is what
  // gets explained (a chart window click), instead of the default (window 0).
  if (tsWindowIndex !== undefined) params.set("ts_window_index", String(tsWindowIndex));

  const res = await fetch(`${BASE}/explain?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Failed to submit job");
  }
  const data = await res.json();
  return data.job_id;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Failed to fetch job status");
  return res.json();
}

export async function cancelExplainJob(jobId: string): Promise<void> {
  const res = await fetch(`${BASE}/jobs/${jobId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to cancel job");
}
