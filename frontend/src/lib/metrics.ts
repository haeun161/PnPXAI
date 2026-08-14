import { ExplainerResult, TaskType } from "./types";

/** One row per PnPXAI metric the backend actually computes.
 *
 * Each metric is shown on its own, under the XAI property it measures — MuFidelity and
 * AbPC are different properties (fidelity of a masked forward pass vs. correctness of the
 * ranked attribution), so averaging them into a single "Faithfulness" number hid one of
 * the two. `tasks` mirrors the per-task metric list in backend/core/pipeline.py: text
 * skips MuFidelity (it masks a 2D pixel grid), and time-series skips both
 * classification-only metrics.
 */
export interface MetricDef {
  key: string;
  label: string;
  field: "mu_fidelity" | "abpc" | "sensitivity" | "complexity";
  tasks: TaskType[];
}

export const METRIC_DEFS: MetricDef[] = [
  { key: "faithfulness", label: "Faithfulness", field: "mu_fidelity", tasks: ["image"] },
  { key: "correctness",  label: "Correctness",  field: "abpc",        tasks: ["image", "text"] },
  { key: "sensitivity",  label: "Robustness",   field: "sensitivity", tasks: ["image", "text", "timeseries"] },
  { key: "complexity",   label: "Compactness",  field: "complexity",  tasks: ["image", "text", "timeseries"] },
];

export const DEFAULT_METRIC_WEIGHTS: Record<string, number> =
  Object.fromEntries(METRIC_DEFS.map((m) => [m.key, 1]));

export function metricsForTask(task: TaskType | ""): MetricDef[] {
  return METRIC_DEFS.filter((m) => task !== "" && m.tasks.includes(task));
}

/** True when the backend could not produce this metric for this explainer.
 *
 * It happens per (explainer, metric) pair rather than per task — e.g. LIME/KernelSHAP
 * raise on Sensitivity/Complexity, and LRP's Sensitivity comes back NaN. Such a metric is
 * counted as 0 (see getMetricValues) but still marked in the UI, so a 0 that means
 * "unavailable" stays distinguishable from a 0 that was measured.
 */
export function isMetricMissing(r: ExplainerResult, def: MetricDef): boolean {
  return r[def.field] == null;
}

/** Metric values for ranking/display, with anything the backend could not compute as 0. */
export function getMetricValues(r: ExplainerResult, task: TaskType | ""): Record<string, number> {
  return Object.fromEntries(metricsForTask(task).map((m) => [m.key, r[m.field] ?? 0]));
}
