"use client";
import { useState, useEffect, useRef } from "react";
import { TaskType, ExplainerInfo } from "@/lib/types";
import TaskSelector from "@/components/TaskSelector";
import ModelSelector from "@/components/ModelSelector";
import DataInput from "@/components/DataInput";
import { getExplainers } from "@/lib/api";
import NavBar from "@/components/NavBar";
import PredictionInfo from "@/components/PredictionInfo";

const BASE = "/api";

interface ParamDef {
  name: string;
  type: string;
  default: any;
}

interface OptResult {
  record_id: string;
  task: string;
  model_name: string;
  explainer_name: string;
  default_params: Record<string, any>;
  optimized_params: Record<string, any>;
  default_metrics: Record<string, number | null>;
  optimized_metrics: Record<string, number | null>;
  available_params: ParamDef[];
  predictions: { class_name: string; probability: number }[];
  visualization_url: string | null;
}

interface CustomResult {
  custom_params: Record<string, any>;
  metrics: Record<string, number | null>;
  visualization_url: string | null;
}

interface HistoryRecord {
  record_id: string;
  task: string;
  model_name: string;
  explainer_name: string;
  timestamp: number;
  optimized_metrics: Record<string, number | null>;
  optimized_params: Record<string, any>;
  available_params: ParamDef[];
  predictions: { class_name: string; probability: number }[];
  visualization_url: string | null;
}

function computeDisplayMetrics(metrics: Record<string, number | null>, taskType: string) {
  const { mu_fidelity, abpc, sensitivity, complexity } = metrics;
  let faithfulness: number | null;
  if (taskType === "text" || taskType === "timeseries") {
    faithfulness = abpc ?? null;
  } else {
    if (mu_fidelity != null && abpc != null) faithfulness = (mu_fidelity + abpc) / 2;
    else faithfulness = mu_fidelity ?? abpc ?? null;
  }
  return [
    { label: "Faithfulness", value: faithfulness },
    { label: "Robustness",   value: sensitivity != null ? -sensitivity : null },
    { label: "Compactness",  value: complexity  != null ? -complexity  : null },
  ];
}

function MetricDisplay({ metrics, taskType }: { metrics: Record<string, number | null>; taskType: string }) {
  const items = computeDisplayMetrics(metrics, taskType);
  return (
    <div className="grid grid-cols-3 gap-1.5">
      {items.map(({ label, value }) => (
        <div key={label} className="bg-gray-50 rounded px-2 py-1 text-center">
          <p className="text-[9px] text-gray-400">{label}</p>
          <p className="text-xs font-mono font-medium text-gray-700">
            {value != null ? Number(value).toFixed(4) : "N/A"}
          </p>
        </div>
      ))}
    </div>
  );
}

export default function OptimizerPage() {
  const [task, setTask] = useState<TaskType | "">("");
  const [model, setModel] = useState("");
  const [explainer, setExplainer] = useState("");
  const [explainers, setExplainers] = useState<ExplainerInfo[]>([]);
  const [inputFile, setInputFile] = useState<File | Blob | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [sourceDataUrl, setSourceDataUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [customLoading, setCustomLoading] = useState(false);
  const [restoredRecordId, setRestoredRecordId] = useState<string | null>(null);

  const [optResult, setOptResult] = useState<OptResult | null>(null);
  const [customResult, setCustomResult] = useState<CustomResult | null>(null);
  const [customParams, setCustomParams] = useState<Record<string, any>>({});
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);
  const autoRunRef = useRef(false);
  const autoRunParamsRef = useRef<{ task: string; model: string; explainer: string } | null>(null);

  useEffect(() => {
    if (!task || !model) return;
    getExplainers(task, model).then(setExplainers).catch(console.error);
  }, [task, model]);

  useEffect(() => {
    fetch(`${BASE}/optimizer/history`).then((r) => r.json()).then(setHistory).catch(() => {});
  }, []);

  // Pre-fill from URL params (coming from ResultCard "Optimize" button)
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const t = params.get("task");
    const m = params.get("model");
    const e = params.get("explainer");
    const dataUrl = params.get("data_url");
    if (t) setTask(t as TaskType);
    if (m) setModel(m);
    if (e) setExplainer(e);
    if (dataUrl && t && m && e) {
      autoRunRef.current = true;
      autoRunParamsRef.current = { task: t, model: m, explainer: e };
      setSourceDataUrl(dataUrl);
      fetch(dataUrl)
        .then((r) => r.blob())
        .then((blob) => {
          setInputFile(blob);
          if (t === "image") setInputPreview(URL.createObjectURL(blob));
        })
        .catch(console.error);
    }
  }, []);

  // Auto-run when coming from ResultCard "Optimize" button
  useEffect(() => {
    if (!autoRunRef.current || !inputFile || !autoRunParamsRef.current) return;
    const { task: t, model: m, explainer: e } = autoRunParamsRef.current;
    if (!t || !m || !e) return;
    autoRunRef.current = false;
    autoRunParamsRef.current = null;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", inputFile);
    const params = new URLSearchParams({ task: t, model_name: m, explainer_name: e, metric_name: "AbPC", n_trials: "20" });
    fetch(`${BASE}/optimizer/optimize?${params}`, { method: "POST", body: formData })
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text();
          try { const err = JSON.parse(text); throw new Error(err.detail || "Optimization failed"); }
          catch { throw new Error(`Server error (${res.status}): ${text.slice(0, 300)}`); }
        }
        return res.json() as Promise<OptResult>;
      })
      .then((data) => {
        setOptResult(data);
        setCustomParams({ ...data.optimized_params });
        fetch(`${BASE}/optimizer/history`).then((r) => r.json()).then(setHistory).catch(() => {});
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed"))
      .finally(() => setLoading(false));
  }, [inputFile]);

  const handleTaskChange = (t: TaskType) => {
    setTask(t);
    setModel("");
    setExplainer("");
    setOptResult(null);
    setCustomResult(null);
    setRestoredRecordId(null);
  };

  const handleOptimize = async () => {
    if (!task || !model || !explainer || !inputFile) return;
    setLoading(true);
    setError(null);
    setOptResult(null);
    setCustomResult(null);
    setRestoredRecordId(null);

    const formData = new FormData();
    formData.append("file", inputFile);
    const params = new URLSearchParams({ task, model_name: model, explainer_name: explainer, metric_name: "AbPC", n_trials: "20" });

    try {
      const res = await fetch(`${BASE}/optimizer/optimize?${params}`, { method: "POST", body: formData });
      if (!res.ok) {
        const text = await res.text();
        try { const e = JSON.parse(text); throw new Error(e.detail || "Optimization failed"); }
        catch { throw new Error(`Server error (${res.status}): ${text.slice(0, 300)}`); }
      }
      const data: OptResult = await res.json();
      setOptResult(data);
      setCustomParams({ ...data.optimized_params });
      fetch(`${BASE}/optimizer/history`).then((r) => r.json()).then(setHistory).catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCustomRun = async () => {
    if (!optResult) return;
    setCustomError(null);

    // If restored from history, use the saved data on server
    if (restoredRecordId) {
      setCustomLoading(true);
      const params = new URLSearchParams({
        explainer_name: optResult.explainer_name,
        custom_params: JSON.stringify(customParams),
      });
      try {
        const res = await fetch(`${BASE}/optimizer/history/${restoredRecordId}/custom?${params}`, { method: "POST" });
        if (!res.ok) {
          const text = await res.text();
          try { const e = JSON.parse(text); throw new Error(e.detail || "Failed"); }
          catch { throw new Error(`Server error (${res.status}): ${text.slice(0, 300)}`); }
        }
        setCustomResult(await res.json());
      } catch (e) {
        setCustomError(e instanceof Error ? e.message : "Failed");
      } finally {
        setCustomLoading(false);
      }
      return;
    }

    // Normal flow with uploaded file
    if (!inputFile) return;
    setCustomLoading(true);
    const formData = new FormData();
    formData.append("file", inputFile);
    const params = new URLSearchParams({
      task: optResult.task, model_name: optResult.model_name, explainer_name: optResult.explainer_name,
      custom_params: JSON.stringify(customParams),
    });

    try {
      const res = await fetch(`${BASE}/optimizer/custom?${params}`, { method: "POST", body: formData });
      if (!res.ok) {
        const text = await res.text();
        try { const e = JSON.parse(text); throw new Error(e.detail || "Failed"); }
        catch { throw new Error(`Server error (${res.status}): ${text.slice(0, 300)}`); }
      }
      setCustomResult(await res.json());
    } catch (e) {
      setCustomError(e instanceof Error ? e.message : "Failed");
    } finally {
      setCustomLoading(false);
    }
  };

  const buildDownloadName = (params: Record<string, any>, explainerName: string) => {
    const paramStr = Object.entries(params)
      .map(([k, v]) => `${k}${v}`)
      .join("_");
    return `${explainerName}${paramStr ? "_" + paramStr : ""}.png`;
  };

  const handleRestoreHistory = async (record: HistoryRecord) => {
    setError(null);
    setCustomResult(null);
    setRestoredRecordId(record.record_id);
    setTask(record.task as TaskType);
    setModel(record.model_name);
    setExplainer(record.explainer_name);
    setOptResult({
      record_id: record.record_id,
      task: record.task,
      model_name: record.model_name,
      explainer_name: record.explainer_name,
      default_params: record.optimized_params || {},
      optimized_params: record.optimized_params || {},
      default_metrics: record.optimized_metrics || {},
      optimized_metrics: record.optimized_metrics || {},
      available_params: record.available_params || [],
      predictions: record.predictions || [],
      visualization_url: record.visualization_url || null,
    });
    setCustomParams({ ...(record.optimized_params || {}) });
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      <main className="flex-1 overflow-hidden max-w-[1600px] w-full mx-auto px-6 py-6">
        <div className="flex gap-6 h-full">
          {/* Left Panel */}
          <div className="w-72 flex-shrink-0 flex flex-col gap-4 overflow-y-auto pr-1">
            <TaskSelector selected={task} onSelect={handleTaskChange} disabled={loading} />

            {task && !restoredRecordId && (
              <>
                <ModelSelector task={task} selected={model} onSelect={(m) => { setModel(m); setExplainer(""); }} disabled={loading} />

                <DataInput task={task} model={model} onDataReady={(data, preview) => { setInputFile(data); setInputPreview(preview); }} disabled={loading} />

                {model && (
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-700">Select Explainer</label>
                    <select
                      value={explainer}
                      onChange={(e) => setExplainer(e.target.value)}
                      disabled={loading}
                      className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="">Choose an explainer...</option>
                      {explainers.filter((e) => e.compatible).map((e) => (
                        <option key={e.name} value={e.name}>{e.display_name}</option>
                      ))}
                    </select>
                  </div>
                )}

                <button
                  onClick={handleOptimize}
                  disabled={!task || !model || !explainer || !inputFile || loading}
                  className="w-full py-2.5 rounded-lg font-semibold text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? "Optimizing..." : "Run Optimization"}
                </button>
              </>
            )}

            {restoredRecordId && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-xs text-green-700">
                <p className="font-semibold mb-1">Restored from history</p>
                <p>{explainer} / {model}</p>
                <button
                  onClick={() => { setRestoredRecordId(null); setOptResult(null); setCustomResult(null); }}
                  className="mt-2 text-green-600 hover:text-green-800 underline"
                >
                  Start new optimization
                </button>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>
            )}

            {/* History */}
            {history.length > 0 && (
              <div className="flex flex-col gap-2 min-h-0">
                <div className="flex items-center justify-between flex-shrink-0">
                  <h3 className="text-sm font-semibold text-gray-700">History</h3>
                  <button
                    onClick={async () => {
                      const toDelete = history.filter((h) => h.record_id !== restoredRecordId);
                      await Promise.all(toDelete.map((h) => fetch(`${BASE}/optimizer/history/${h.record_id}`, { method: "DELETE" })));
                      setHistory((prev) => prev.filter((h) => h.record_id === restoredRecordId));
                    }}
                    className="text-[10px] text-gray-400 hover:text-red-500 transition-colors"
                  >
                    Clear all
                  </button>
                </div>
                <div className="overflow-y-auto flex flex-col gap-2 pr-0.5">
                  {history.map((h, i) => (
                    <div
                      key={h.record_id || i}
                      className={`relative bg-gray-50 hover:bg-blue-50 rounded-lg p-2.5 text-xs transition-colors border ${
                        restoredRecordId === h.record_id ? "border-blue-400 bg-blue-50" : "border-transparent"
                      }`}
                    >
                      <button
                        onClick={() => handleRestoreHistory(h)}
                        className="w-full text-left"
                      >
                        <div className="font-medium text-gray-700">{h.explainer_name}</div>
                        <div className="text-gray-400">{h.task} / {h.model_name}</div>
                        <div className="text-gray-400">{new Date(h.timestamp * 1000).toLocaleString()}</div>
                        <div className="text-blue-500 mt-1">Click to resume →</div>
                      </button>
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          await fetch(`${BASE}/optimizer/history/${h.record_id}`, { method: "DELETE" });
                          setHistory((prev) => prev.filter((r) => r.record_id !== h.record_id));
                          if (restoredRecordId === h.record_id) {
                            setRestoredRecordId(null);
                            setOptResult(null);
                            setCustomResult(null);
                          }
                        }}
                        className="absolute top-2 right-2 text-gray-300 hover:text-red-500 transition-colors"
                        title="Delete record"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="flex-1 overflow-y-auto">
            {!optResult && !loading && (
              <div className="text-center py-20 text-gray-400">
                <svg className="mx-auto h-12 w-12 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <p className="text-sm">Select task, data, model, and explainer, then click &quot;Run Optimization&quot;</p>
                <p className="text-xs mt-1">Or click a recent session to resume</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center py-20 gap-6">
                {/* Concentric spinning rings */}
                <div className="relative w-20 h-20 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-green-200 border-r-green-200 animate-spin" style={{ animationDuration: "3s" }} />
                  <div className="absolute inset-[5px] rounded-full border-2 border-transparent border-t-green-400 border-l-green-400 animate-spin" style={{ animationDuration: "1.5s", animationDirection: "reverse" }} />
                  <div className="absolute inset-[10px] rounded-full border-2 border-transparent border-t-green-600 animate-spin" style={{ animationDuration: "0.75s" }} />
                  <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                </div>

                {/* Equalizer bars */}
                <div className="flex items-end gap-px" style={{ height: "28px" }}>
                  {[45, 85, 35, 100, 60, 75, 40, 90, 55, 70, 50, 80].map((h, i) => (
                    <div
                      key={i}
                      className="w-1.5 rounded-t-sm animate-bounce"
                      style={{
                        height: `${h}%`,
                        backgroundColor: `rgba(34, 197, 94, ${0.2 + i * 0.065})`,
                        animationDelay: `${i * 65}ms`,
                        animationDuration: `${520 + (i % 4) * 130}ms`,
                      }}
                    />
                  ))}
                </div>

                <div className="text-center">
                  <p className="text-sm font-medium text-gray-700">Running Optimization</p>
                  <p className="text-xs text-gray-400 mt-1">Searching for the best parameters...</p>
                </div>
              </div>
            )}

            {optResult && (
              <div className="space-y-4">
                <PredictionInfo
                  dataUrl={sourceDataUrl ?? inputPreview}
                  predictions={optResult.predictions}
                  task={optResult.task as TaskType}
                />

                <div className="grid grid-cols-2 gap-4">
                  {/* Left: Optimized - FIXED */}
                  <div className="bg-white rounded-xl border-2 border-green-200 p-4 space-y-3">
                    <h3 className="text-sm font-semibold text-green-700 flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Optimized Result
                    </h3>
                    {optResult.visualization_url && (
                      <div className="bg-gray-50 rounded-lg overflow-hidden relative group">
                        <img src={optResult.visualization_url} alt="Optimized XAI" className="w-full max-h-96 object-contain" />
                        <a
                          href={optResult.visualization_url}
                          download={buildDownloadName(optResult.optimized_params, optResult.explainer_name)}
                          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 rounded-md px-2 py-1 text-[10px] text-gray-600 hover:text-gray-900 shadow-sm flex items-center gap-1"
                        >
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                          Download
                        </a>
                      </div>
                    )}
                    <div className="grid grid-cols-2 divide-x divide-gray-100 border border-gray-100 rounded-lg overflow-hidden">
                      <div className="p-2">
                        <p className="text-[10px] font-semibold text-gray-500 mb-1">PARAMETERS</p>
                        <div className="space-y-0.5">
                          {Object.entries(optResult.optimized_params).length === 0 && (
                            <p className="text-[10px] text-gray-400 italic">None</p>
                          )}
                          {Object.entries(optResult.optimized_params).map(([k, v]) => (
                            <div key={k} className="flex justify-between text-xs gap-2">
                              <span className="text-gray-500 truncate">{k}</span>
                              <span className="font-mono text-gray-700 flex-shrink-0">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="p-2">
                        <p className="text-[10px] font-semibold text-gray-500 mb-1">METRICS</p>
                        <MetricDisplay metrics={optResult.optimized_metrics} taskType={optResult.task} />
                      </div>
                    </div>
                  </div>

                  {/* Right: Custom - UPDATES */}
                  <div className="bg-white rounded-xl border-2 border-blue-200 p-4 space-y-3">
                    <h3 className="text-sm font-semibold text-blue-700 flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                      Custom Parameters
                    </h3>
                    {customResult?.visualization_url ? (
                      <div className="bg-gray-50 rounded-lg overflow-hidden relative group">
                        <img src={customResult.visualization_url} alt="Custom XAI" className="w-full max-h-96 object-contain" />
                        <a
                          href={customResult.visualization_url}
                          download={buildDownloadName(customParams, optResult.explainer_name)}
                          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 rounded-md px-2 py-1 text-[10px] text-gray-600 hover:text-gray-900 shadow-sm flex items-center gap-1"
                        >
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                          Download
                        </a>
                      </div>
                    ) : (
                      <div className="bg-gray-50 rounded-lg h-48 flex items-center justify-center text-xs text-gray-400">
                        Adjust parameters and click &quot;Run&quot; to see XAI result
                      </div>
                    )}
                    <div className="grid grid-cols-2 divide-x divide-gray-100 border border-gray-100 rounded-lg overflow-hidden">
                      <div className="p-2 space-y-1.5">
                        <div className="flex items-center justify-between">
                          <p className="text-[10px] font-semibold text-gray-500">PARAMETERS</p>
                          <button
                            onClick={handleCustomRun}
                            disabled={customLoading}
                            className="px-2 py-0.5 rounded text-[10px] font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 transition-colors"
                          >
                            {customLoading ? "Running..." : "Run"}
                          </button>
                        </div>
                        {(optResult.available_params || []).length === 0 && (
                          <p className="text-[10px] text-gray-400 italic">None</p>
                        )}
                        {(optResult.available_params || []).map((p) => (
                          <div key={p.name}>
                            <div className="flex justify-between">
                              <label className="text-[10px] text-gray-500">{p.name}</label>
                              <span className="text-[9px] text-gray-400">default: {String(p.default)}</span>
                            </div>
                            <input
                              type={p.type === "int" || p.type === "float" ? "number" : "text"}
                              step={p.type === "float" ? "0.01" : p.type === "int" ? "1" : undefined}
                              value={customParams[p.name] ?? p.default ?? ""}
                              onChange={(e) => {
                                const val = p.type === "int" ? parseInt(e.target.value) : p.type === "float" ? parseFloat(e.target.value) : e.target.value;
                                setCustomParams((prev) => ({ ...prev, [p.name]: val }));
                              }}
                              className="w-full rounded border border-gray-300 px-2 py-1 text-xs font-mono focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            />
                          </div>
                        ))}
                        {customError && (
                          <div className="bg-red-50 border border-red-200 rounded-lg px-2 py-1.5 text-[10px] text-red-700 leading-snug">
                            {customError}
                          </div>
                        )}
                      </div>
                      <div className="p-2">
                        <p className="text-[10px] font-semibold text-gray-500 mb-1">METRICS</p>
                        {customResult ? (
                          <MetricDisplay metrics={customResult.metrics} taskType={optResult.task} />
                        ) : (
                          <p className="text-[10px] text-gray-400">Run to see metrics</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
