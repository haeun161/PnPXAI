"use client";
import { useEffect, useRef, useState } from "react";
import { ExplainerResult, JobStatus, TaskType } from "@/lib/types";
import ResultCard from "./ResultCard";
import ProgressIndicator from "./ProgressIndicator";

interface Props {
  results: ExplainerResult[];
  task: TaskType;
  rankingMetric: string;
  job: JobStatus | null;
  loading: boolean;
  hiddenExplainers?: string[];
  metricWeights?: Record<string, number>;
  onWeightChange?: (metric: string, value: number) => void;
  onResetWeights?: () => void;
  className?: string;
}

function getFaithfulness(r: ExplainerResult, task: TaskType): number | null {
  if (task === "text" || task === "timeseries") return r.abpc;
  if (r.mu_fidelity != null && r.abpc != null) return (r.mu_fidelity + r.abpc) / 2;
  return r.mu_fidelity ?? r.abpc;
}

type MetricMap = Record<string, number | null>;

function getMetricValues(r: ExplainerResult, task: TaskType): MetricMap {
  return {
    faithfulness: getFaithfulness(r, task),
    sensitivity: r.sensitivity,
    complexity: r.complexity,
  };
}

function computeMinMax(completed: ExplainerResult[], task: TaskType): Record<string, { min: number; max: number }> {
  const keys = ["faithfulness", "sensitivity", "complexity"];
  const bounds: Record<string, { min: number; max: number }> = {};
  for (const key of keys) {
    const vals = completed
      .map((r) => getMetricValues(r, task)[key])
      .filter((v): v is number => v != null);
    bounds[key] = {
      min: vals.length > 0 ? Math.min(...vals) : 0,
      max: vals.length > 0 ? Math.max(...vals) : 1,
    };
  }
  return bounds;
}

function getRankScore(
  r: ExplainerResult,
  weights: Record<string, number>,
  task: TaskType,
  bounds: Record<string, { min: number; max: number }>
): number {
  const values = getMetricValues(r, task);
  let sum = 0, total = 0;
  for (const [key, w] of Object.entries(weights)) {
    if (w <= 0) continue;
    const val = values[key] ?? null;
    if (val == null) continue;
    const { min, max } = bounds[key] ?? { min: 0, max: 1 };
    const normalized = max > min ? (val - min) / (max - min) : 1;
    sum += normalized * w;
    total += w;
  }
  return total > 0 ? sum / total : 0;
}

function rerank(results: ExplainerResult[], weights: Record<string, number>, task: TaskType): ExplainerResult[] {
  const completed = results.filter((r) => r.status === "completed").map((r) => ({ ...r }));
  const bounds = computeMinMax(completed, task);
  completed.sort((a, b) => getRankScore(b, weights, task, bounds) - getRankScore(a, weights, task, bounds));
  completed.forEach((r, i) => { r.rank = i + 1; });
  return [...completed, results.filter((r) => r.status !== "completed")].flat();
}

const METRIC_LABELS = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "sensitivity",  label: "Robustness" },
  { key: "complexity",   label: "Compactness" },
];

const DEFAULT_WEIGHTS: Record<string, number> = {
  faithfulness: 1, sensitivity: 1, complexity: 1,
};

interface WeightControlsProps {
  metricWeights: Record<string, number>;
  onWeightChange: (metric: string, value: number) => void;
  onResetWeights: () => void;
}

function WeightControls({ metricWeights, onWeightChange, onResetWeights }: WeightControlsProps) {
  const [gearOpen, setGearOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editingVal, setEditingVal] = useState("");
  const gearRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!gearOpen) return;
    const handler = (e: MouseEvent) => {
      if (gearRef.current && !gearRef.current.contains(e.target as Node)) setGearOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [gearOpen]);

  const activeCount = METRIC_LABELS.filter(({ key }) => (metricWeights[key] ?? 0) > 0).length;
  const total = METRIC_LABELS.reduce((s, { key }) => s + (metricWeights[key] ?? 0), 0);

  return (
    <div className="flex items-center gap-2">
      {/* 3 toggle buttons */}
      <div className="flex gap-1">
        {METRIC_LABELS.map(({ key, label }) => {
          const active = (metricWeights[key] ?? 0) > 0;
          const isLast = active && activeCount === 1;
          return (
            <button
              key={key}
              onClick={() => { if (!isLast) onWeightChange(key, active ? 0 : 1); }}
              disabled={isLast}
              title={isLast ? "At least one metric required" : undefined}
              className={`text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all ${
                active
                  ? isLast
                    ? "bg-blue-50 border-blue-200 text-blue-400 cursor-not-allowed"
                    : "bg-blue-50 border-blue-300 text-blue-600"
                  : "bg-white border-gray-200 text-gray-400 hover:border-blue-200 hover:text-blue-400"
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Gear */}
      <div ref={gearRef} className="relative">
        <button
          onClick={() => setGearOpen((o) => !o)}
          className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${
            gearOpen ? "text-blue-500 bg-blue-50" : "text-gray-400 hover:text-blue-500 hover:bg-gray-100"
          }`}
          title="Adjust weights"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>

        {gearOpen && (
          <div className="absolute right-0 top-8 z-30 w-56 bg-white border border-gray-200 rounded-xl shadow-lg p-3">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Adjust Weights</span>
              <button onClick={onResetWeights} className="text-[10px] text-gray-400 hover:text-blue-600 transition-colors">reset</button>
            </div>
            <div className="flex flex-col gap-2.5">
              {METRIC_LABELS.map(({ key, label }) => {
                const w = metricWeights[key] ?? 0;
                const pct = total > 0 ? Math.round((w / total) * 100) : 0;
                return (
                  <div key={key} className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span className={`text-[11px] flex-1 truncate ${w > 0 ? "text-gray-600" : "text-gray-300"}`}>{label}</span>
                      <span className={`text-[10px] font-mono tabular-nums w-7 text-right ${w > 0 ? "text-blue-500" : "text-gray-300"}`}>
                        {w > 0 ? `${pct}%` : "—"}
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => onWeightChange(key, Math.max(0, Math.round((w - 0.1) * 10) / 10))}
                          disabled={w <= 0.1 && activeCount === 1}
                          className="w-5 h-5 rounded border border-gray-200 text-gray-400 hover:text-gray-700 hover:border-gray-300 flex items-center justify-center text-xs leading-none transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                        >−</button>
                        {editingKey === key ? (
                          <input
                            autoFocus
                            type="number"
                            min={0} max={9} step={0.1}
                            value={editingVal}
                            onChange={(e) => setEditingVal(e.target.value)}
                            onBlur={() => {
                              const parsed = parseFloat(editingVal);
                              const clamped = Math.round(Math.min(9, Math.max(0, isNaN(parsed) ? w : parsed)) * 10) / 10;
                              if (!(clamped === 0 && activeCount === 1)) onWeightChange(key, clamped);
                              setEditingKey(null);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                              if (e.key === "Escape") setEditingKey(null);
                            }}
                            className="w-8 text-center text-xs font-mono border border-blue-300 rounded outline-none bg-white text-gray-800"
                          />
                        ) : (
                          <span
                            onClick={() => { setEditingKey(key); setEditingVal(String(w)); }}
                            className={`w-8 text-center text-xs font-mono tabular-nums cursor-text rounded hover:bg-gray-100 px-0.5 ${w > 0 ? "text-gray-700" : "text-gray-300"}`}
                          >
                            {w.toFixed(1)}
                          </span>
                        )}
                        <button
                          onClick={() => onWeightChange(key, Math.min(9, Math.round((w + 0.1) * 10) / 10))}
                          className="w-5 h-5 rounded border border-gray-200 text-gray-400 hover:text-gray-700 hover:border-gray-300 flex items-center justify-center text-xs leading-none transition-colors"
                        >+</button>
                      </div>
                    </div>
                    <div className="h-1 rounded-full bg-gray-100 overflow-hidden">
                      <div className="h-full rounded-full bg-blue-300 transition-all duration-300" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsPanel({ results, task, job, loading, hiddenExplainers = [], metricWeights = DEFAULT_WEIGHTS, onWeightChange, onResetWeights, className }: Props) {
  const [expanded, setExpanded] = useState(false);

  const handleWeightChange = onWeightChange ?? (() => {});
  const handleResetWeights = onResetWeights ?? (() => {});

  const activeMetrics = Object.entries(metricWeights)
    .filter(([, w]) => w > 0)
    .map(([k]) => k);

  const METRIC_DISPLAY: Record<string, string> = {
    faithfulness: "Faithfulness", sensitivity: "Robustness", complexity: "Compactness",
  };
  const rankLabel = activeMetrics.length === 0
    ? "no metrics"
    : activeMetrics.length >= 3
    ? "weighted avg"
    : activeMetrics.map((m) => METRIC_DISPLAY[m] ?? m).join(", ");

  if (results.length === 0) {
    return (
      <div>
        <ProgressIndicator job={job} loading={loading} />
        {!loading && (
          <div className="text-center py-12 text-gray-400">
            <svg className="mx-auto h-12 w-12 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm">Results will appear here after analysis</p>
          </div>
        )}
      </div>
    );
  }

  const rankedResults = rerank(results, metricWeights, task)
    .filter((r) => r.status !== "failed")
    .filter((r) => !hiddenExplainers.includes(r.explainer_name));

  if (expanded) {
    return (
      <div className="fixed inset-0 z-50 bg-white overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between z-10">
          <h3 className="text-base font-semibold text-gray-800">
            Attribution Results
            <span className="font-normal text-gray-400 ml-1 text-sm">(ranked by {rankLabel})</span>
          </h3>
          <div className="flex items-center gap-4">
            <WeightControls
              metricWeights={metricWeights}
              onWeightChange={handleWeightChange}
              onResetWeights={handleResetWeights}
            />
            <button
              onClick={() => setExpanded(false)}
              className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 border border-gray-300 rounded-lg px-3 py-1.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Close
            </button>
          </div>
        </div>
        <div className="p-6">
          <div className="mb-4"><ProgressIndicator job={job} loading={loading} /></div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {rankedResults.map((r) => (
              <ResultCard key={r.explainer_name} result={r} task={task} activeMetrics={activeMetrics} modelName={job?.model_name} dataUrl={job?.original_data_url} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col${className ? ` ${className}` : ""}`}>
      <ProgressIndicator job={job} loading={loading} />
      <div className={`flex items-center justify-between mb-2 ${loading || job ? "mt-3" : ""}`}>
        <h3 className="text-sm font-semibold text-gray-700">
          Attribution Results
          <span className="font-normal text-gray-400 ml-1">(ranked by {rankLabel})</span>
        </h3>
        <button
          onClick={() => setExpanded(true)}
          className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 border border-blue-200 rounded-md px-2 py-0.5"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
          </svg>
          Expand ({rankedResults.length})
        </button>
      </div>
      <div className={`flex gap-3 overflow-x-auto pb-2 ${task === "text" ? "h-[490px]" : task === "timeseries" ? "flex-1 min-h-0" : "h-100"}`}>
        {rankedResults.map((r) => (
          <div key={r.explainer_name} className="flex-shrink-0 h-full" style={{ width: task === "timeseries" ? "calc((100% - 0.75rem) / 2)" : "calc((100% - 2.25rem) / 4)" }}>
            <ResultCard result={r} task={task} activeMetrics={activeMetrics} modelName={job?.model_name} dataUrl={job?.original_data_url} />
          </div>
        ))}
      </div>
    </div>
  );
}
