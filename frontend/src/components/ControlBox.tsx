"use client";
import { useEffect, useRef, useState } from "react";
import { ExplainerResult, JobStatus, TaskType } from "@/lib/types";

interface Props {
  explainers: string[];
  displayNames: Record<string, string>;
  job: JobStatus | null;
  loading: boolean;
  hiddenExplainers: string[];
  onToggleHidden: (name: string) => void;
  onSetAllHidden: (hidden: boolean) => void;
  task?: TaskType;
  metricWeights: Record<string, number>;
  onWeightChange: (metric: string, value: number) => void;
  onResetWeights: () => void;
}

const STATUS_STYLES: Record<string, { block: string; dot: string }> = {
  completed:     { block: "border-green-300 bg-green-50 text-green-700",  dot: "bg-green-500" },
  running:       { block: "border-blue-300 bg-blue-50 text-blue-700",    dot: "bg-blue-500 animate-pulse" },
  pending:       { block: "border-gray-200 bg-gray-50 text-gray-500",    dot: "bg-gray-300" },
  failed:        { block: "border-red-200 bg-red-50 text-red-500",       dot: "bg-red-400" },
  not_supported: { block: "border-gray-100 bg-white text-gray-400",      dot: "bg-gray-200" },
};

const METRIC_LABELS: { key: string; label: string }[] = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "sensitivity",  label: "Robustness" },
  { key: "complexity",   label: "Compactness" },
];

export default function ControlBox({
  explainers, displayNames, job, loading,
  hiddenExplainers, onToggleHidden, onSetAllHidden,
  task, metricWeights, onWeightChange, onResetWeights,
}: Props) {
  const resultMap = new Map<string, ExplainerResult>();
  job?.results.forEach((r) => resultMap.set(r.explainer_name, r));

  const completedCount = job?.results.filter((r) => r.status === "completed").length ?? 0;
  const total = explainers.length;
  const allHidden = explainers.length > 0 && explainers.every((n) => hiddenExplainers.includes(n));

  const [gearOpen, setGearOpen] = useState(false);
  const gearRef = useRef<HTMLDivElement>(null);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editingVal, setEditingVal] = useState("");

  useEffect(() => {
    if (!gearOpen) return;
    const handler = (e: MouseEvent) => {
      if (gearRef.current && !gearRef.current.contains(e.target as Node)) setGearOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [gearOpen]);

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm space-y-3">
      {/* Explainers header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">Explainers</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onSetAllHidden(!allHidden)}
            className="text-[10px] text-gray-400 hover:text-blue-600 transition-colors"
          >
            {allHidden ? "Show all" : "Hide all"}
          </button>
          <span className={`text-[10px] font-medium tabular-nums ${loading ? "text-blue-600" : "text-gray-400"}`}>
            {completedCount}/{total}
          </span>
        </div>
      </div>

      {/* Explainer list — sorted by rank, scrollable */}
      <div className="flex flex-col gap-1 max-h-24 overflow-y-auto pr-0.5">
        {[...explainers]
          .sort((a, b) => {
            const ra = resultMap.get(a)?.rank ?? 999;
            const rb = resultMap.get(b)?.rank ?? 999;
            return ra - rb;
          })
          .map((name) => {
            const result = resultMap.get(name);
            const status = result?.status ?? "pending";
            const displayName = displayNames[name] ?? name;
            const styles = STATUS_STYLES[status] ?? STATUS_STYLES.pending;
            const isHidden = hiddenExplainers.includes(name);
            return (
              <button
                key={name}
                onClick={() => onToggleHidden(name)}
                className={`flex items-center gap-2 text-[11px] font-medium px-2 py-1.5 rounded-lg border transition-all text-left ${
                  isHidden ? "opacity-40 line-through border-gray-200 bg-gray-50 text-gray-400" : styles.block
                }`}
                title={isHidden ? "Click to show" : (result?.current_step ?? status)}
              >
                {result?.rank != null && (
                  <span className="text-[9px] font-bold text-gray-400 w-4 text-center flex-shrink-0">
                    #{result.rank}
                  </span>
                )}
                <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isHidden ? "bg-gray-300" : styles.dot}`} />
                <span className="truncate">{displayName}</span>
              </button>
            );
          })}
      </div>

      {/* Divider */}
      <div className="border-t border-gray-100" />

      {/* Ranking weights */}
      <div>
        {/* Header row */}
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">Ranking Metric</span>
          <div ref={gearRef} className="relative">
            <button
              onClick={() => setGearOpen((o) => !o)}
              className={`w-5 h-5 flex items-center justify-center rounded transition-colors ${
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

            {/* Weight adjustment popover */}
            {gearOpen && (
              <div className="absolute right-0 bottom-7 z-30 w-56 bg-white border border-gray-200 rounded-xl shadow-lg p-3">
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">Adjust Weights</span>
                  <button
                    onClick={onResetWeights}
                    className="text-[10px] text-gray-400 hover:text-blue-600 transition-colors"
                  >
                    reset
                  </button>
                </div>
                {(() => {
                  const activeCount = METRIC_LABELS.filter(({ key }) => (metricWeights[key] ?? 0) > 0).length;
                  const total = METRIC_LABELS.reduce((s, { key }) => s + (metricWeights[key] ?? 0), 0);
                  return (
                    <div className="flex flex-col gap-2.5">
                      {METRIC_LABELS.map(({ key, label }) => {
                        const w = metricWeights[key] ?? 0;
                        const pct = total > 0 ? Math.round((w / total) * 100) : 0;
                        return (
                          <div key={key} className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                              <span className={`text-[11px] flex-1 truncate ${w > 0 ? "text-gray-600" : "text-gray-300"}`}>
                                {label}
                              </span>
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
                                    min={0}
                                    max={9}
                                    step={0.1}
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
                            {/* proportion bar */}
                            <div className="h-1 rounded-full bg-gray-100 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-blue-300 transition-all duration-300"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                      {total === 0 && (
                        <p className="text-[10px] text-gray-400 text-center mt-1">Select at least one metric</p>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </div>

        {/* 3 metric toggle buttons */}
        {(() => {
          const activeCount = METRIC_LABELS.filter(({ key }) => (metricWeights[key] ?? 0) > 0).length;
          return (
            <div className="flex gap-1.5">
              {METRIC_LABELS.map(({ key, label }) => {
                const active = (metricWeights[key] ?? 0) > 0;
                const isLast = active && activeCount === 1;
                return (
                  <button
                    key={key}
                    onClick={() => { if (!isLast) onWeightChange(key, active ? 0 : 1); }}
                    disabled={isLast}
                    title={isLast ? "At least one metric required" : undefined}
                    className={`flex-1 text-[10px] font-semibold py-1.5 rounded-lg border transition-all ${
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
          );
        })()}
      </div>
    </div>
  );
}
