"use client";
import { useEffect, useRef, useState } from "react";
import { ExplainerInfo, TaskType } from "@/lib/types";
import { getExplainers } from "@/lib/api";

interface RankedResult {
  name: string;
  display_name: string;
  estimated_compute_time_seconds: number;
  metrics: Record<string, number | null>;
  avg_score: number;
}

interface DetectJob {
  status: "running" | "completed" | "error";
  current: number;
  total: number;
  current_explainer: string;
  detected_architectures: string[];
  results: RankedResult[];
  error: string | null;
}

export interface DetectionCache {
  state: "idle" | "running" | "completed" | "error";
  job: DetectJob | null;
  selected: string[];
  error: string | null;
}

interface Props {
  task: TaskType;
  model: string;
  inputData: File | Blob | null;
  cache: DetectionCache;
  onCacheChange: (c: DetectionCache) => void;
  onSave: (selected: string[]) => void;
  onClose: () => void;
}

type State = "idle" | "running" | "completed" | "error";

const CATEGORY_MAP: Record<string, string> = {
  Lime: "Perturbation", KernelShap: "Perturbation",
  LRPUniformEpsilon: "Relevance", RAP: "Relevance",
  GradCam: "CAM", GuidedGradCam: "CAM",
  Gradient: "Gradient", GradientXInput: "Gradient",
  IntegratedGradients: "Gradient", SmoothGrad: "Gradient", VarGrad: "Gradient",
};

const CATEGORY_COLORS: Record<string, string> = {
  Perturbation: "bg-purple-100 text-purple-700",
  Relevance: "bg-amber-100 text-amber-700",
  CAM: "bg-green-100 text-green-700",
  Gradient: "bg-blue-100 text-blue-700",
};

const ARCH_COLORS: Record<string, string> = {
  Linear: "bg-sky-100 text-sky-700 border-sky-200",
  Convolution: "bg-violet-100 text-violet-700 border-violet-200",
  Attention: "bg-rose-100 text-rose-700 border-rose-200",
  RNN: "bg-teal-100 text-teal-700 border-teal-200",
  LSTM: "bg-teal-100 text-teal-700 border-teal-200",
  Embedding: "bg-orange-100 text-orange-700 border-orange-200",
  Pool: "bg-gray-100 text-gray-600 border-gray-200",
};

export default function ExplainerDetectionModal({ task, model, inputData, cache, onCacheChange, onSave, onClose }: Props) {
  const [state, setStateRaw] = useState<State>(cache.state);
  const [job, setJobRaw] = useState<DetectJob | null>(cache.job);
  const [selected, setSelectedRaw] = useState<string[]>(cache.selected);
  const [error, setErrorRaw] = useState<string | null>(cache.error);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const setState = (s: State) => { setStateRaw(s); onCacheChange({ ...cache, state: s }); };
  const setJob = (j: DetectJob | null) => { setJobRaw(j); onCacheChange({ ...cache, job: j }); };
  const setSelected = (sel: string[] | ((prev: string[]) => string[])) => {
    setSelectedRaw((prev) => {
      const next = typeof sel === "function" ? sel(prev) : sel;
      onCacheChange({ ...cache, selected: next });
      return next;
    });
  };
  const setError = (e: string | null) => { setErrorRaw(e); onCacheChange({ ...cache, error: e }); };

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const runDetection = async () => {
    if (!inputData) {
      setError("Please upload input data first.");
      return;
    }
    setState("running");
    setError(null);
    setJob(null);

    try {
      const formData = new FormData();
      formData.append("file", inputData);
      const params = new URLSearchParams({ task, model_name: model });
      const res = await fetch(`/api/detect-rank?${params}`, { method: "POST", body: formData });
      if (!res.ok) {
        const e = await res.json().catch(() => ({ detail: "Detection failed" }));
        throw new Error(e.detail || "Detection failed");
      }
      const { job_id } = await res.json();

      // Poll until done
      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`/api/detect-rank/${job_id}`);
          const data: DetectJob = await r.json();
          setJobRaw(data);
          if (data.status === "completed") {
            clearInterval(pollRef.current!);
            const sel = data.results.slice(0, 5).map((r) => r.name);
            setSelectedRaw(sel);
            setStateRaw("completed");
            onCacheChange({ state: "completed", job: data, selected: sel, error: null });
          } else if (data.status === "error") {
            clearInterval(pollRef.current!);
            const msg = data.error || "Detection failed";
            setErrorRaw(msg);
            setStateRaw("error");
            onCacheChange({ state: "error", job: data, selected: [], error: msg });
          } else {
            onCacheChange({ state: "running", job: data, selected: [], error: null });
          }
        } catch {
          clearInterval(pollRef.current!);
          setErrorRaw("Polling failed");
          setStateRaw("error");
          onCacheChange({ state: "error", job: null, selected: [], error: "Polling failed" });
        }
      }, 1500);
    } catch (e: any) {
      setError(e.message || "Detection failed");
      setState("error");
    }
  };

  const toggle = (name: string) =>
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );

  const formatTime = (s: number) => s < 10 ? `~${s}s` : `~${Math.round(s)}s`;

  const handleRedetect = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    const reset: DetectionCache = { state: "idle", job: null, selected: [], error: null };
    setStateRaw("idle"); setJobRaw(null); setSelectedRaw([]); setErrorRaw(null);
    onCacheChange(reset);
  };

  const pct = job && job.total > 0 ? Math.round((job.current / job.total) * 100) : 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl w-[520px] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-gray-900">Explainer Detection</h2>
              <p className="text-xs text-gray-400">Evaluate and rank all compatible XAI methods</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 flex flex-col overflow-y-auto" style={{ minHeight: 280, maxHeight: "65vh" }}>

          {/* Idle */}
          {state === "idle" && (
            <div className="flex flex-col items-center justify-center flex-1 text-center py-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-700 mb-1">Run all compatible XAI methods</p>
              <p className="text-xs text-gray-400 mb-1 leading-relaxed">
                Evaluates MuFidelity, AbPC, Sensitivity, Complexity<br />
                for each method and ranks by average score.
              </p>
              {!inputData && (
                <p className="text-xs text-amber-500 mb-4">⚠ Upload input data first</p>
              )}
              {inputData && <div className="mb-4" />}
              <button
                onClick={runDetection}
                disabled={!inputData}
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white text-sm font-semibold px-6 py-2.5 rounded-xl transition-colors shadow-sm"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Run Detection
              </button>
            </div>
          )}

          {/* Running */}
          {state === "running" && (
            <div className="flex flex-col gap-4">
              {/* Architecture badges */}
              {job?.detected_architectures && job.detected_architectures.length > 0 && (
                <div className="bg-gray-50 rounded-xl px-4 py-3 border border-gray-100">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="w-3.5 h-3.5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Detected Architectures</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {job.detected_architectures.map((arch) => (
                      <span key={arch} className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                        {arch}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Progress */}
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-500 font-medium">
                    {job ? `Evaluating ${job.current} / ${job.total}` : "Starting..."}
                  </span>
                  <span className="text-blue-600 font-semibold">{pct}%</span>
                </div>
                <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                {job?.current_explainer && (
                  <p className="text-xs text-gray-400 mt-1.5 flex items-center gap-1.5">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                    Running: <span className="font-medium text-gray-600">{job.current_explainer}</span>
                  </p>
                )}
              </div>

              {/* Partial results */}
              {job && job.results.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Completed so far</p>
                  <div className="space-y-1.5">
                    {[...job.results].sort((a, b) => b.avg_score - a.avg_score).map((r) => (
                      <div key={r.name} className="flex items-center gap-2 text-xs py-1.5 px-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-700 font-medium flex-1">{r.display_name}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${CATEGORY_COLORS[CATEGORY_MAP[r.name] ?? ""] ?? "bg-gray-100 text-gray-500"}`}>
                          {CATEGORY_MAP[r.name] ?? "Other"}
                        </span>
                        <span className="font-mono text-gray-500 w-12 text-right">{r.avg_score.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {state === "error" && (
            <div className="flex flex-col items-center justify-center flex-1 text-center py-6">
              <div className="w-12 h-12 rounded-xl bg-red-50 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                </svg>
              </div>
              <p className="text-sm text-red-600 mb-3">{error}</p>
              <button onClick={handleRedetect} className="text-sm text-blue-600 hover:text-blue-700 font-medium">Try again</button>
            </div>
          )}

          {/* Completed */}
          {state === "completed" && job && (
            <div className="flex flex-col gap-4">
              {/* Architecture */}
              {job.detected_architectures.length > 0 && (
                <div className="bg-gray-50 rounded-xl px-4 py-3 border border-gray-100">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="w-3.5 h-3.5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Detected Architectures</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {job.detected_architectures.map((arch) => (
                      <span key={arch} className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                        {arch}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Top 5 ranked results */}
              <div>
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    Top {Math.min(5, job.results.length)} · ranked by avg score
                  </span>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${selected.length > 0 ? "bg-blue-50 text-blue-600" : "bg-gray-100 text-gray-400"}`}>
                    {selected.length} selected
                  </span>
                </div>
                <div className="space-y-2">
                  {job.results.slice(0, 5).map((r, idx) => {
                    const isSelected = selected.includes(r.name);
                    const category = CATEGORY_MAP[r.name] ?? "Other";
                    const colorClass = CATEGORY_COLORS[category] ?? "bg-gray-100 text-gray-600";
                    return (
                      <label
                        key={r.name}
                        className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 cursor-pointer transition-all ${
                          isSelected ? "border-blue-300 bg-blue-50/60" : "border-gray-200 bg-white hover:border-gray-300"
                        }`}
                      >
                        <div className="w-5 h-5 rounded-full bg-gray-100 text-gray-500 text-[10px] font-bold flex items-center justify-center flex-shrink-0">
                          {idx + 1}
                        </div>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggle(r.name)}
                          className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 flex-shrink-0"
                        />
                        <span className={`text-sm font-medium flex-1 ${isSelected ? "text-blue-800" : "text-gray-700"}`}>
                          {r.display_name}
                        </span>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${colorClass}`}>{category}</span>
                          <span className="text-xs font-mono text-gray-500 w-12 text-right">{r.avg_score.toFixed(3)}</span>
                        </div>
                      </label>
                    );
                  })}
                </div>
                <button
                  onClick={handleRedetect}
                  className="mt-3 text-xs text-gray-400 hover:text-gray-500 flex items-center gap-1 transition-colors"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Re-detect
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 flex items-center justify-between">
          <span className="text-xs text-gray-400">
            {state === "completed" && selected.length === 0 && "Select at least 1 method"}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="text-sm text-gray-500 hover:text-gray-700 px-4 py-2 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
            >
              Cancel
            </button>
            {state === "completed" && (
              <button
                onClick={() => { onSave(selected); onClose(); }}
                disabled={selected.length === 0}
                className="text-sm font-semibold bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
                Save Selection
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
