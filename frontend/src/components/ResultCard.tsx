"use client";
import { useState } from "react";
import { ExplainerResult, TaskType } from "@/lib/types";

interface Props {
  result: ExplainerResult;
  task: TaskType;
  activeMetrics: string[];
}

const METRIC_LABELS: Record<string, string> = {
  mu_fidelity: "Accuracy (Fidelity)",
  abpc: "Accuracy (AbPC)",
  sensitivity: "Sensitivity",
  complexity: "Complexity",
};

export default function ResultCard({ result, task, activeMetrics }: Props) {
  const [showExpanded, setShowExpanded] = useState(false);
  const isCompleted = result.status === "completed";
  const isNotSupported = result.status === "not_supported";
  const isFailed = result.status === "failed";
  const isRunning = result.status === "running" || result.status === "pending";

  const allMetrics = [
    { key: "mu_fidelity", label: "Accuracy (Fidelity)", value: result.mu_fidelity },
    { key: "abpc", label: "Accuracy (AbPC)", value: result.abpc },
    { key: "sensitivity", label: "Sensitivity", value: result.sensitivity },
    { key: "complexity", label: "Complexity", value: result.complexity },
  ];
  const metrics = task === "text" ? allMetrics.filter((m) => m.key !== "mu_fidelity") : allMetrics;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Visualization */}
      <div className="relative bg-gray-50 aspect-square">
        {result.rank != null && (
          <div className="absolute top-2 left-2 z-10 w-7 h-7 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center shadow">
            #{result.rank}
          </div>
        )}

        {isCompleted && result.visualization_url && (
          <>
            <img src={result.visualization_url} alt={result.display_name} className="w-full h-full object-contain" />
            {task === "timeseries" && (
              <button
                onClick={() => setShowExpanded(true)}
                className="absolute bottom-2 right-2 bg-white/90 hover:bg-white rounded-md px-2 py-1 text-[10px] text-gray-600 hover:text-blue-700 shadow-sm flex items-center gap-1 transition-colors"
                title="Show all variables"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
                </svg>
                Expand
              </button>
            )}
          </>
        )}
        {/* Expanded modal for all timeseries variables */}
        {showExpanded && result.visualization_url && (
          <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center" onClick={() => setShowExpanded(false)}>
            <div className="bg-white rounded-xl shadow-xl max-w-3xl max-h-[85vh] overflow-auto p-4" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-800">{result.display_name} — All Variables</h3>
                <button onClick={() => setShowExpanded(false)} className="text-gray-400 hover:text-gray-600">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <img
                src={result.visualization_url.replace(".png", "_expanded.png")}
                alt={`${result.display_name} expanded`}
                className="w-full object-contain"
              />
            </div>
          </div>
        )}
        {/* Token highlighting for text tasks */}
        {isCompleted && task === "text" && result.token_attributions && (
          <div className="absolute bottom-0 left-0 right-0 bg-white/90 p-2 max-h-[50%] overflow-y-auto">
            <div className="flex flex-wrap gap-0.5">
              {result.token_attributions.map((ta, i) => (
                <span
                  key={i}
                  className="px-0.5 py-px rounded text-[10px]"
                  style={{
                    backgroundColor: `rgba(239, 68, 68, ${Math.min(ta.score, 1) * 0.7})`,
                    color: ta.score > 0.5 ? "white" : "inherit",
                  }}
                  title={`${ta.token}: ${ta.score.toFixed(3)}`}
                >
                  {ta.token}
                </span>
              ))}
            </div>
          </div>
        )}
        {isNotSupported && (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 p-4">
            <svg className="w-8 h-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <span className="text-xs text-center font-medium text-red-500">Not Supported</span>
          </div>
        )}
        {isFailed && (
          <div className="w-full h-full flex flex-col items-center justify-center text-red-400 p-4">
            <svg className="w-8 h-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-xs text-center">Error</span>
            {result.error_message && (
              <span className="text-xs text-center text-gray-400 mt-1 line-clamp-2">{result.error_message}</span>
            )}
          </div>
        )}
        {isRunning && (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-gray-400 mt-2">Computing...</span>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-3">
        <h4 className="text-sm font-semibold text-gray-800">{result.display_name}</h4>
        {isCompleted && (
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-center">
            {metrics.map((m) => {
              const isRanked = activeMetrics.includes(m.key);
              return (
                <div key={m.key} className={isRanked ? "bg-blue-50 rounded px-1" : ""}>
                  <p className="text-[10px] text-gray-400">{m.label}</p>
                  <p className="text-xs font-mono font-medium text-gray-700">
                    {m.value?.toFixed(4) ?? "N/A"}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
