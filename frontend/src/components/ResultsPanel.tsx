"use client";
import { useEffect, useRef, useState } from "react";
import { ExplainerResult, JobStatus, TaskType } from "@/lib/types";
import {
  DEFAULT_METRIC_WEIGHTS, MetricDef, getMetricValues, metricsForTask,
} from "@/lib/metrics";
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

// context_labels are "YYYY-MM-DD HH:MM" (backend/tasks/timeseries.py); the explained
// window may be any chained segment now (a chart window click), not just the first, so
// this always reflects whichever one `job.forecast.context` actually describes.
function parseContextLabel(s: string) {
  const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})$/.exec(s);
  return m ? { yyyy: m[1], mm: m[2], dd: m[3], hh: m[4], mi: m[5] } : null;
}

function labelToMinutes(p: { yyyy: string; mm: string; dd: string; hh: string; mi: string }): number {
  return Date.UTC(Number(p.yyyy), Number(p.mm) - 1, Number(p.dd), Number(p.hh), Number(p.mi)) / 60000;
}

// The end date drops the year when it matches the start's, since a context window is at
// most a few days long and repeating it just adds noise.
function formatContextRange(labels: string[] | null | undefined): string | null {
  if (!labels || labels.length === 0) return null;
  const start = parseContextLabel(labels[0]);
  const end = parseContextLabel(labels[labels.length - 1]);
  if (!start || !end) return `${labels[0]} ~ ${labels[labels.length - 1]}`;
  const startStr = `${start.yyyy}/${start.mm}/${start.dd} ${start.hh}:${start.mi}`;
  const endStr = end.yyyy === start.yyyy
    ? `${end.mm}/${end.dd} ${end.hh}:${end.mi}`
    : `${end.yyyy}/${end.mm}/${end.dd} ${end.hh}:${end.mi}`;
  return `${startStr} ~ ${endStr}`;
}

// "지난 N시간/일/분" -- seq_len (the number of context points) times the sampling
// interval between them, read off the labels themselves so it's right regardless of
// the data's actual granularity (hourly, daily, ...).
function formatContextSpan(labels: string[] | null | undefined): string | null {
  if (!labels || labels.length < 2) return null;
  const p0 = parseContextLabel(labels[0]);
  const p1 = parseContextLabel(labels[1]);
  if (!p0 || !p1) return null;
  const stepMin = labelToMinutes(p1) - labelToMinutes(p0);
  if (stepMin <= 0) return null;
  const totalMin = stepMin * labels.length;
  if (totalMin % 60 !== 0) return `지난 ${totalMin}분`;
  const hours = totalMin / 60;
  // Beyond ~2 days, "N일" reads better than a large hour count -- but a clean 24h
  // window (a very common seq_len) should still read as "지난 24시간", not "1일".
  if (hours % 24 === 0 && hours > 48) return `지난 ${hours / 24}일`;
  return `지난 ${hours}시간`;
}

// Small metadata pill for the explained-input date range: set off from the heading
// with its own border/background so it reads as data, not as part of the title, with
// tabular-nums keeping the two timestamps aligned.
function InputRangeBadge({ range, span, className = "" }: { range: string; span?: string | null; className?: string }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border border-gray-200 bg-gray-50 px-3 py-1.5 text-gray-600 ${className}`}
    >
      <svg className="w-4 h-4 flex-shrink-0 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span className="font-medium text-gray-500 whitespace-nowrap">Input window:</span>
      <span className="tabular-nums whitespace-nowrap font-semibold text-gray-800">{range}</span>
      {span && (
        <>
          <span className="text-gray-300">·</span>
          <span className="whitespace-nowrap text-gray-500">{span}</span>
        </>
      )}
    </span>
  );
}

/** Weighted average over the metrics available for the task.
 *
 * A metric the backend could not compute counts as 0 (getMetricValues), rather than being
 * dropped from the average — dropping it used to give an explainer with no metrics at all
 * an empty average, i.e. a score of 0, which beat every real (sign-flipped, hence
 * negative) Robustness/Compactness score and floated it to rank 1.
 */
function getRankScore(
  r: ExplainerResult,
  weights: Record<string, number>,
  task: TaskType,
): number {
  const values = getMetricValues(r, task);
  let sum = 0, total = 0;
  for (const { key } of metricsForTask(task)) {
    const w = weights[key] ?? 0;
    if (w <= 0) continue;
    sum += values[key] * w;
    total += w;
  }
  return total > 0 ? sum / total : 0;
}

// A text card is a 380px token heatmap stacked on the title plus one row per metric
// (~184px for three), and the strip's own pb-2 takes another 8px. Tailwind's JIT only sees
// literal class names, so this is applied as an inline height, not an h-[...] class.
const TEXT_CARD_HEIGHT = 580;

function rerank(results: ExplainerResult[], weights: Record<string, number>, task: TaskType): ExplainerResult[] {
  const completed = results.filter((r) => r.status === "completed").map((r) => ({ ...r }));
  completed.sort((a, b) => getRankScore(b, weights, task) - getRankScore(a, weights, task));
  completed.forEach((r, i) => { r.rank = i + 1; });
  return [...completed, results.filter((r) => r.status !== "completed")].flat();
}

interface WeightControlsProps {
  metricWeights: Record<string, number>;
  onWeightChange: (metric: string, value: number) => void;
  onResetWeights: () => void;
  metricDefs: MetricDef[];
}

function WeightControls({ metricWeights, onWeightChange, onResetWeights, metricDefs }: WeightControlsProps) {
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

  const activeCount = metricDefs.filter(({ key }) => (metricWeights[key] ?? 0) > 0).length;
  const total = metricDefs.reduce((s, { key }) => s + (metricWeights[key] ?? 0), 0);

  return (
    <div className="flex items-center gap-2">
      {/* One toggle per metric available for the task */}
      <div className="flex gap-1">
        {metricDefs.map(({ key, label }) => {
          const active = (metricWeights[key] ?? 0) > 0;
          const isLast = active && activeCount === 1;
          return (
            <button
              key={key}
              onClick={() => { if (!isLast) onWeightChange(key, active ? 0 : 1); }}
              disabled={isLast}
              title={isLast ? "At least one metric required" : undefined}
              className={`flex items-center gap-1 text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all ${
                active
                  ? isLast
                    ? "bg-blue-50 border-blue-200 text-blue-400 cursor-not-allowed"
                    : "bg-blue-50 border-blue-300 text-blue-600"
                  : "bg-white border-gray-200 text-gray-400 hover:border-blue-200 hover:text-blue-400"
              }`}
            >
              {label}
              <svg className="w-2.5 h-2.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 19V5M5 12l7-7 7 7" />
              </svg>
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
              {metricDefs.map(({ key, label }) => {
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

export default function ResultsPanel({ results, task, job, loading, hiddenExplainers = [], metricWeights = DEFAULT_METRIC_WEIGHTS, onWeightChange, onResetWeights, className }: Props) {
  const [expanded, setExpanded] = useState(false);

  const handleWeightChange = onWeightChange ?? (() => {});
  const handleResetWeights = onResetWeights ?? (() => {});

  // Only the metrics the backend computes for this task can be ranked on or labelled —
  // text has no MuFidelity, time-series neither MuFidelity nor AbPC.
  const metricDefs = metricsForTask(task);
  const activeMetrics = metricDefs
    .filter(({ key }) => (metricWeights[key] ?? 0) > 0)
    .map(({ key }) => key);

  const rankLabel = activeMetrics.length === 0
    ? "no metrics"
    : activeMetrics.length === metricDefs.length && metricDefs.length > 1
    ? "weighted avg"
    : metricDefs.filter(({ key }) => activeMetrics.includes(key)).map(({ label }) => label).join(", ");

  // Only show completed/failed/not_supported as real cards — pending ones are not yet visible
  const visibleResults = results
    .filter((r) => !hiddenExplainers.includes(r.explainer_name))
    .filter((r) => r.status === "completed" || r.status === "failed" || r.status === "not_supported");

  // Show a single placeholder for the currently running (or first pending) explainer
  const currentExplainerName = loading
    ? (results.find((r) => r.status === "running") ?? results.find((r) => r.status === "pending"))?.explainer_name
    : undefined;
  const currentPlaceholder: ExplainerResult[] = currentExplainerName && !hiddenExplainers.includes(currentExplainerName)
    ? [{
        explainer_name: currentExplainerName,
        display_name: currentExplainerName,
        status: "pending" as const,
        rank: null, visualization_url: null,
        mu_fidelity: null, abpc: null, sensitivity: null, complexity: null,
        not_supported_reason: null, error_message: null,
        token_attributions: null, current_step: null,
      }]
    : [];

  const rankedResults = [
    ...rerank(visibleResults, metricWeights, task).filter((r) => r.status !== "failed"),
    ...currentPlaceholder,
  ];

  const inputRange = task === "timeseries" ? formatContextRange(job?.forecast?.context_labels) : null;
  const inputSpan = task === "timeseries" ? formatContextSpan(job?.forecast?.context_labels) : null;

  if (expanded) {
    return (
      <div className="fixed inset-0 z-50 bg-white overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between z-10">
          <div className="flex items-center gap-3 min-w-0">
            <h3 className="text-lg font-semibold text-gray-800 flex-shrink-0">
              Explanation Results
              <span className="font-normal text-gray-400 ml-1 text-base">(ranked by {rankLabel})</span>
            </h3>
            {inputRange && <InputRangeBadge range={inputRange} span={inputSpan} className="text-lg" />}
          </div>
          <div className="flex items-center gap-4">
            <WeightControls
              metricWeights={metricWeights}
              onWeightChange={handleWeightChange}
              onResetWeights={handleResetWeights}
              metricDefs={metricDefs}
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
          <div className="p-6 overflow-y-auto" style={{ height: "calc(100vh - 61px)" }}>
          {/* A time-series attribution is a wide strip chart, so it gets three per row
              where an image gets five — a fifth of the width turned it into a tall narrow
              column that object-contain then letterboxed into a sliver. Both wrap onto the
              next row, with rows sized so two of them fill the viewport — except text,
              which needs its own full card height (half a viewport clipped the metrics). */}
          <div
            className={`grid gap-3 ${task === "timeseries" || task === "text" ? "grid-cols-3" : "grid-cols-5"}`}
            style={{ gridAutoRows: task === "text" ? `${TEXT_CARD_HEIGHT}px` : "calc((100vh - 61px - 48px - 16px) / 2)" }}
          >
            {rankedResults.map((r, i) => (
              <div key={r.explainer_name} className="animate-card-in h-full" style={{ animationDelay: `${i * 60}ms` }}>
                <ResultCard result={r} task={task} activeMetrics={activeMetrics} metricWeights={metricWeights} modelName={job?.model_name} dataUrl={job?.original_data_url} isExpanded />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const cardContainerClass = task === "text" ? "" : task === "timeseries" ? "flex-1 min-h-0" : "h-[485px]";
  const cardContainerStyle = task === "text" ? { height: TEXT_CARD_HEIGHT } : undefined;
  // How many cards fit before the strip scrolls horizontally. Text gets 3: a token
  // heatmap needs the width to stay readable, and there is now one metric row per
  // metric under it. Time-series strips are wider still, so they get 2.
  const visibleCards = task === "timeseries" ? 2 : task === "text" ? 3 : 4;
  const cardWidth = `calc((100% - ${0.75 * (visibleCards - 1)}rem) / ${visibleCards})`;

  return (
    <div className={`flex flex-col${className ? ` ${className}` : ""}`}>
      <ProgressIndicator job={job} loading={loading} />
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <h3 className="text-base font-semibold text-gray-700 flex-shrink-0">
            Explanation Results
          </h3>
          {inputRange && <InputRangeBadge range={inputRange} span={inputSpan} className="text-base" />}
        </div>
        <button
          onClick={() => setExpanded(true)}
          className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 border border-blue-200 rounded-md px-2 py-0.5"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
          </svg>
          Expand ({rankedResults.length})
        </button>
      </div>
      <div className={`flex gap-3 overflow-x-auto pb-2 ${cardContainerClass}`} style={cardContainerStyle}>
        {rankedResults.map((r, i) => (
          <div key={r.explainer_name} className="animate-card-in flex-shrink-0 h-full" style={{ width: cardWidth, animationDelay: `${i * 60}ms` }}>
            <ResultCard result={r} task={task} activeMetrics={activeMetrics} metricWeights={metricWeights} modelName={job?.model_name} dataUrl={job?.original_data_url} />
          </div>
        ))}
      </div>
    </div>
  );
}
