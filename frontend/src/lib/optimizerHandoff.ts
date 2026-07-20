import { PredictionItem } from "./types";

// In-memory hand-off from the Explanation page to the Optimizer page for the
// "Optimize" button on a ResultCard. Client-side navigation (next/link) keeps
// the JS runtime alive, so we can pass the actual uploaded File/Blob and the
// job's predictions directly instead of re-fetching them over the network —
// that network round-trip is what silently failed whenever the backend was
// briefly down or the source job had expired.
interface OptimizerHandoff {
  dataUrl: string;
  file: File | Blob;
  predictions: PredictionItem[] | null;
}

let pending: OptimizerHandoff | null = null;

export function setOptimizerHandoff(data: OptimizerHandoff) {
  pending = data;
}

export function getOptimizerHandoff(): OptimizerHandoff | null {
  return pending;
}
