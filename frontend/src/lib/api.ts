import { TaskInfo, TaskType, ModelInfo, ExplainerInfo, JobStatus } from "./types";

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

export async function submitExplainJob(
  task: TaskType,
  file: File | Blob,
  modelName: string,
  explainerNames: string[],
  rankingMetric: string = "average",
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({
    task,
    model_name: modelName,
    explainer_names: explainerNames.join(","),
    ranking_metric: rankingMetric,
  });

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
