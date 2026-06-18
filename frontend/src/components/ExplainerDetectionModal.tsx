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
  current_step: string;
  detected_architectures: string[];
  results: RankedResult[];
  error: string | null;
  linked_job_id?: string;
}

export interface DetectionCache {
  state: "idle" | "running" | "completed" | "error";
  job: DetectJob | null;
  selected: string[];
  error: string | null;
  linkedJobId?: string;
}

interface Props {
  task: TaskType;
  model: string;
  inputData: File | Blob | null;
  cache: DetectionCache;
  onCacheChange: (c: DetectionCache) => void;
  onGo: (selected: string[]) => void;
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


const PIPELINE_STEPS = [
  { label: "Attribution",   keys: ["attribution"] },
  { label: "Faithfulness",  keys: ["mu_fidelity", "abpc"] },
  { label: "Robustness",    keys: ["sensitivity"] },
  { label: "Compactness",   keys: ["complexity"] },
  { label: "Visualization", keys: ["visualization"] },
];

const MIN_STEP_MS = 280;

const ARCH_COLORS: Record<string, string> = {
  Linear: "bg-sky-100 text-sky-700 border-sky-200",
  Convolution: "bg-violet-100 text-violet-700 border-violet-200",
  Attention: "bg-rose-100 text-rose-700 border-rose-200",
  RNN: "bg-teal-100 text-teal-700 border-teal-200",
  LSTM: "bg-teal-100 text-teal-700 border-teal-200",
  Embedding: "bg-orange-100 text-orange-700 border-orange-200",
  Pool: "bg-gray-100 text-gray-600 border-gray-200",
};


export default function ExplainerDetectionModal({ task, model, inputData, cache, onCacheChange, onGo, onClose }: Props) {
  const [state, setStateRaw] = useState<State>(cache.state);
  const [job, setJobRaw] = useState<DetectJob | null>(cache.job);
  const [selected, setSelectedRaw] = useState<string[]>(cache.selected);
  const [error, setErrorRaw] = useState<string | null>(cache.error);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Pipeline step timing queue
  const [displayedStepIdx, setDisplayedStepIdx] = useState(-1);
  const displayedIdxRef = useRef(-1);
  const stepQueueRef = useRef<number[]>([]);
  const processingStepsRef = useRef(false);
  const lastExplainerRef = useRef("");
  const processStepsRef = useRef<() => void>(() => {});
  processStepsRef.current = () => {
    if (stepQueueRef.current.length === 0) { processingStepsRef.current = false; return; }
    processingStepsRef.current = true;
    const next = stepQueueRef.current.shift()!;
    if (next === -1) {
      // Reset sentinel: clear display for new explainer, then continue immediately
      displayedIdxRef.current = -1;
      setDisplayedStepIdx(-1);
      setTimeout(() => processStepsRef.current(), 80);
    } else {
      displayedIdxRef.current = next;
      setDisplayedStepIdx(next);
      setTimeout(() => processStepsRef.current(), MIN_STEP_MS);
    }
  };


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

  // Drive pipeline step display with minimum visibility time per step.
  // -1 in the queue is a reset sentinel marking the boundary between explainers.
  // pendingNewExplainerStepRef tracks how far the new explainer has progressed
  // while its steps are still queued behind the sentinel.
  const pendingNewStepRef = useRef(-1);

  useEffect(() => {
    if (!job) return;
    const explainer = job.current_explainer ?? "";
    const activeIdx = PIPELINE_STEPS.findIndex((s) => s.keys.includes(job.current_step ?? ""));

    if (explainer !== lastExplainerRef.current && explainer !== "") {
      // Finish previous explainer's remaining steps, then sentinel, then start new from 0
      const last = displayedIdxRef.current;
      for (let i = last + 1; i < PIPELINE_STEPS.length; i++) stepQueueRef.current.push(i);
      stepQueueRef.current.push(-1); // boundary sentinel
      lastExplainerRef.current = explainer;
      pendingNewStepRef.current = -1; // reset new-explainer cursor
    }

    if (activeIdx < 0) return;

    const hasSentinel = stepQueueRef.current.includes(-1);
    if (hasSentinel) {
      // New explainer steps go after the sentinel; track via pendingNewStepRef
      const from = pendingNewStepRef.current + 1;
      for (let i = from; i <= activeIdx; i++) stepQueueRef.current.push(i);
      pendingNewStepRef.current = activeIdx;
    } else {
      // Normal within-explainer progression
      const from = displayedIdxRef.current + 1;
      for (let i = from; i <= activeIdx; i++) stepQueueRef.current.push(i);
    }

    if (!processingStepsRef.current) processStepsRef.current();
  }, [job?.current_step, job?.current_explainer]);

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
            const sel = data.results.map((r) => r.name);
            setSelectedRaw(sel);
            setStateRaw("completed");
            onCacheChange({ state: "completed", job: data, selected: sel, error: null, linkedJobId: data.linked_job_id });
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
      onClick={state === "running" ? undefined : onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl w-[600px] flex flex-col overflow-hidden"
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
          <button onClick={onClose} disabled={state === "running"} className="text-gray-400 hover:text-gray-600 transition-colors disabled:opacity-30 disabled:cursor-not-allowed">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 flex flex-col overflow-y-auto" style={{ minHeight: 320, maxHeight: "70vh" }}>

          {/* Idle */}
          {state === "idle" && (
            <div className="flex flex-col items-center justify-center flex-1 text-center py-8">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-700 mb-1">Run all compatible XAI methods</p>
              <p className="text-xs text-gray-400 mb-5 leading-relaxed">
                Detects model architecture, then evaluates each compatible<br />
                explainer across Faithfulness · Robustness · Compactness.
              </p>
              {!inputData && <p className="text-xs text-amber-500 mb-4">⚠ Upload input data first</p>}
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
            <div className="flex flex-col gap-5">
              {/* Architecture — appears once detected */}
              {job?.detected_architectures && job.detected_architectures.length > 0 ? (
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mr-1">Architecture</span>
                  {job.detected_architectures.map((arch) => (
                    <span key={arch} className={`text-xs font-medium px-2.5 py-0.5 rounded-lg border ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                      {arch}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  <span className="text-xs text-gray-400">Detecting architecture…</span>
                </div>
              )}

              {/* Pipeline steps */}
              <div className="rounded-2xl border border-blue-100 bg-gradient-to-br from-blue-50 to-indigo-50 px-4 py-4">
                {/* Header */}
                <div className="flex items-center gap-2 mb-4">
                  <div className="relative w-5 h-5 flex-shrink-0">
                    <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-blue-400 animate-spin" style={{ animationDuration: "1s" }} />
                    <div className="absolute inset-[2px] rounded-full border-2 border-transparent border-t-blue-600 animate-spin" style={{ animationDuration: "0.7s", animationDirection: "reverse" }} />
                  </div>
                  <span className="text-xs font-semibold text-blue-700 truncate">{job?.current_explainer ?? "Starting…"}</span>
                </div>

                {/* Steps */}
                <div className="flex items-center">
                  {PIPELINE_STEPS.map((step, i) => {
                    const done = displayedStepIdx > i;
                    const active = displayedStepIdx === i;
                    return (
                      <div key={step.label} className="flex items-center flex-1 min-w-0">
                        {/* Node */}
                        <div className={`relative flex-1 min-w-0 flex flex-col items-center gap-1.5 py-3 px-1 rounded-xl transition-all duration-300 ${
                          active ? "bg-blue-600 shadow-md shadow-blue-200" : done ? "bg-blue-500/20" : ""
                        }`}>
                          {active && <div className="absolute inset-0 rounded-xl ring-2 ring-blue-400 ring-offset-1 animate-pulse" />}
                          {/* Dot */}
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center transition-all duration-300 ${
                            active ? "bg-white/25" : done ? "bg-blue-500" : "bg-gray-200"
                          }`}>
                            {done ? (
                              <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                              </svg>
                            ) : active ? (
                              <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                            ) : (
                              <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
                            )}
                          </div>
                          {/* Label */}
                          <span className={`text-[9px] font-semibold text-center leading-tight transition-colors ${
                            active ? "text-white" : done ? "text-blue-600" : "text-gray-400"
                          }`}>
                            {step.label}
                          </span>
                        </div>
                        {/* Arrow connector */}
                        {i < PIPELINE_STEPS.length - 1 && (
                          <svg className={`w-4 h-4 flex-shrink-0 transition-colors duration-300 ${done ? "text-blue-400" : "text-gray-200"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Overall progress bar */}
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-gray-500">Overall progress</span>
                  <span className="text-blue-600 font-semibold tabular-nums">{pct}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500 transition-all duration-500" style={{ width: `${pct}%` }} />
                </div>
              </div>

              {/* Completed chips */}
              {job && job.results.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-2">
                    Completed ({job.results.length})
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {job.results.map((r) => (
                      <span key={r.name} className="inline-flex items-center gap-1 text-[11px] font-medium px-2.5 py-1 rounded-lg bg-green-50 text-green-700 border border-green-200">
                        <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                        {r.display_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {state === "error" && (
            <div className="flex flex-col items-center justify-center flex-1 text-center py-8">
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
              {/* Architecture + summary row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mr-1">Architecture</span>
                  {job.detected_architectures.map((arch) => (
                    <span key={arch} className={`text-xs font-medium px-2.5 py-0.5 rounded-lg border ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                      {arch}
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${selected.length > 0 ? "bg-blue-50 text-blue-600" : "bg-gray-100 text-gray-400"}`}>
                    {selected.length} / {job.results.length} selected
                  </span>
                  <button
                    onClick={() => setSelected(selected.length === job.results.length ? [] : job.results.map((r) => r.name))}
                    className="text-[10px] text-gray-400 hover:text-blue-600 transition-colors"
                  >
                    {selected.length === job.results.length ? "Deselect all" : "Select all"}
                  </button>
                </div>
              </div>

              {/* Explainer grid — 2 columns, no scores */}
              <div className="grid grid-cols-2 gap-2">
                {job.results.map((r) => {
                  const isSelected = selected.includes(r.name);
                  const category = CATEGORY_MAP[r.name] ?? "Other";
                  const dotColor = {
                    Perturbation: "bg-purple-400",
                    Relevance: "bg-amber-400",
                    CAM: "bg-green-400",
                    Gradient: "bg-blue-400",
                  }[category] ?? "bg-gray-300";
                  return (
                    <label
                      key={r.name}
                      className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 cursor-pointer transition-all ${
                        isSelected ? "border-blue-300 bg-blue-50/70" : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggle(r.name)}
                        className="w-3.5 h-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 flex-shrink-0"
                      />
                      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColor}`} />
                      <span className={`text-sm font-medium truncate ${isSelected ? "text-blue-800" : "text-gray-700"}`}>
                        {r.display_name}
                      </span>
                    </label>
                  );
                })}
              </div>

              <button
                onClick={handleRedetect}
                className="self-start text-xs text-gray-400 hover:text-gray-500 flex items-center gap-1 transition-colors"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Re-detect
              </button>
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
                onClick={() => onGo(selected)}
                disabled={selected.length === 0}
                className="text-sm font-semibold bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                GO
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
