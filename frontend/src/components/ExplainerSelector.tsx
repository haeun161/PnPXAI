"use client";
import { useEffect, useRef, useState } from "react";
import { ExplainerInfo, ExplainerResult, TaskType } from "@/lib/types";
import { getExplainers } from "@/lib/api";

const EXPLAINER_CATEGORIES = [
  {
    label: "Perturbation-based",
    names: ["Lime", "KernelShap"],
  },
  {
    label: "Relevance-based",
    names: ["LRPUniformEpsilon", "RAP"],
  },
  {
    label: "CAM-based",
    names: ["GradCam", "GuidedGradCam"],
  },
  {
    label: "Gradient-based",
    names: ["Gradient", "GradientXInput", "IntegratedGradients", "SmoothGrad", "VarGrad"],
  },
];

interface Props {
  task: TaskType | "";
  model: string;
  selected: string[];
  onSelect: (names: string[]) => void;
  disabled?: boolean;
  results?: ExplainerResult[];
}

export default function ExplainerSelector({ task, model, selected, onSelect, disabled, results }: Props) {
  const [explainers, setExplainers] = useState<ExplainerInfo[]>([]);
  const [open, setOpen] = useState(false);
  const didInit = useRef(false);

  useEffect(() => {
    if (!task || !model) { setExplainers([]); return; }
    didInit.current = false;
    getExplainers(task, model).then((list) => {
      setExplainers(list);
      if (!didInit.current) {
        didInit.current = true;
        onSelect(list.map((e) => e.name));
      }
    }).catch(console.error);
  }, [task, model]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!results || results.length === 0) return;
    const failedNames = results.filter((r) => r.status === "failed").map((r) => r.explainer_name);
    if (failedNames.length === 0) return;
    const next = selected.filter((n) => !failedNames.includes(n));
    if (next.length !== selected.length) onSelect(next);
  }, [results]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (name: string) => {
    if (selected.includes(name)) {
      onSelect(selected.filter((n) => n !== name));
    } else {
      onSelect([...selected, name]);
    }
  };

  const formatTime = (s: number) => (s < 10 ? `~${s}s` : `~${Math.round(s)}s`);

  const resultMap = Object.fromEntries((results ?? []).map((r) => [r.explainer_name, r]));
  const hasResults = (results ?? []).length > 0;

  if (!task || !model) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-gray-700">Explainers Information</label>
        <p className="text-xs text-gray-400">{!task ? "Select a task first" : "Select a model first"}</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-semibold text-gray-700">
          Explainers Information
          <span className="font-normal text-gray-400 ml-1">({selected.length}/{explainers.length})</span>
        </label>
        <button
          onClick={() => setOpen(true)}
          disabled={disabled || explainers.length === 0}
          title="Configure explainers"
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 border border-gray-200 hover:border-gray-300 rounded-md px-2 py-1 transition-colors disabled:opacity-40"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <circle cx="12" cy="12" r="3" strokeWidth={2} />
          </svg>
          Configure
        </button>
      </div>

      {/* Selected tags preview */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.slice(0, 4).map((name) => {
            const exp = explainers.find((e) => e.name === name);
            return (
              <span key={name} className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 rounded px-1.5 py-0.5">
                {exp?.display_name ?? name}
              </span>
            );
          })}
          {selected.length > 4 && (
            <span className="text-[10px] text-gray-400">+{selected.length - 4} more</span>
          )}
        </div>
      )}

      {/* Modal */}
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setOpen(false)}>
          <div
            className="bg-white rounded-2xl shadow-xl w-[1000px] max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
              <div>
                <h2 className="text-sm font-semibold text-gray-900">Configure Explainers</h2>
                <p className="text-xs text-gray-400 mt-0.5">{selected.length} of {explainers.length} selected</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onSelect(explainers.map((e) => e.name))}
                  className="text-xs text-blue-600 hover:text-blue-700"
                >
                  All
                </button>
                <span className="text-gray-300">|</span>
                <button
                  onClick={() => onSelect([])}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  None
                </button>
                <button onClick={() => setOpen(false)} className="ml-2 text-gray-400 hover:text-gray-600">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* 4-column category grid */}
            <div className="overflow-y-auto flex-1 px-4 py-4">
              <div className="grid grid-cols-4 gap-3">
                {EXPLAINER_CATEGORIES.map((cat) => {
                  const catExplainers = explainers.filter((e) => cat.names.includes(e.name));
                  if (catExplainers.length === 0) return null;
                  return (
                    <div key={cat.label}>
                      <p className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2 pb-1.5 border-b border-gray-200 whitespace-nowrap text-center">
                        {cat.label}
                      </p>
                      <div className="space-y-1.5">
                        {catExplainers.map((exp) => {
                          const isSelected = selected.includes(exp.name);
                          const r = resultMap[exp.name];
                          const isFailed = r?.status === "failed";
                          const isCompleted = r?.status === "completed";

                          return (
                            <label
                              key={exp.name}
                              className={`flex flex-col gap-1 rounded-lg border px-2.5 py-2 cursor-pointer transition-colors ${
                                isSelected
                                  ? "border-blue-400 bg-blue-50"
                                  : "border-gray-200 hover:border-gray-300 bg-white"
                              }`}
                            >
                              <div className="flex items-start gap-2">
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={() => toggle(exp.name)}
                                  className="mt-0.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 flex-shrink-0"
                                />
                                <div className="min-w-0">
                                  <p className="text-xs font-medium text-gray-800 leading-snug">{exp.display_name}</p>
                                  <div className="flex items-center gap-1 mt-0.5 flex-wrap">
                                    <span className="text-[10px] text-gray-400">{formatTime(exp.estimated_compute_time_seconds)}</span>
                                    {exp.estimated_compute_time_seconds >= 20 && (
                                      <span className="text-[10px] bg-yellow-100 text-yellow-700 px-1 py-px rounded">slow</span>
                                    )}
                                    {isCompleted && (
                                      <span className="text-[10px] bg-green-100 text-green-700 px-1 py-px rounded flex items-center gap-0.5">
                                        <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                        </svg>
                                        done
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              {/* Failure reason */}
                              {isFailed && r.error_message && (
                                <div className="flex items-start gap-1 text-[10px] text-red-600 bg-red-50 border border-red-100 rounded px-1.5 py-1">
                                  <svg className="w-2.5 h-2.5 flex-shrink-0 mt-px" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                                  </svg>
                                  <span className="leading-snug">{r.error_message}</span>
                                </div>
                              )}
                              {/* Pre-run known incompatibility */}
                              {!hasResults && !exp.compatible && exp.incompatibility_reason && (
                                <p className="text-[10px] text-amber-600 leading-snug">{exp.incompatibility_reason}</p>
                              )}
                            </label>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-gray-100 flex justify-end">
              <button
                onClick={() => setOpen(false)}
                className="text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white px-4 py-1.5 rounded-lg transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
