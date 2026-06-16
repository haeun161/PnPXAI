"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { TaskType } from "@/lib/types";
import TaskSelector from "@/components/TaskSelector";
import DataInput from "@/components/DataInput";
import ModelSelector from "@/components/ModelSelector";
import ExplainerDetectionModal, { DetectionCache } from "@/components/ExplainerDetectionModal";
import ExplainerSelector from "@/components/ExplainerSelector";
import RankingMetricSelector from "@/components/RankingMetricSelector";
import PredictionInfo from "@/components/PredictionInfo";
import ResultsPanel from "@/components/ResultsPanel";
import ControlBox from "@/components/ControlBox";
import { useExplainJob } from "@/hooks/useExplainJob";
import NavBar from "@/components/NavBar";

function WelcomePanel() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center px-8 -mt-[60px]">
      <img src="/pnpxai_logo.png" alt="PnPXAI Logo" className="w-[410px] mb-10 select-none" draggable={false} />
      <h2 className="text-[1.75rem] font-bold text-gray-900 mb-4">PnPXAI: Plug-and-Play Explainable AI</h2>
      <p className="text-[1.05rem] text-gray-500 leading-relaxed max-w-3xl">
        PnPXAI is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI).
        It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way. 
        Select a task, model, and input data on the left to get started.
      </p>
    </div>
  );
}

const ARCH_COLORS: Record<string, string> = {
  Linear: "bg-sky-100 text-sky-700 border-sky-200",
  Convolution: "bg-violet-100 text-violet-700 border-violet-200",
  Attention: "bg-rose-100 text-rose-700 border-rose-200",
  RNN: "bg-teal-100 text-teal-700 border-teal-200",
  LSTM: "bg-teal-100 text-teal-700 border-teal-200",
  Embedding: "bg-orange-100 text-orange-700 border-orange-200",
  Pool: "bg-gray-100 text-gray-600 border-gray-200",
};

const DEFAULT_WEIGHTS = { faithfulness: 1, sensitivity: 1, complexity: 1 };
const SS_KEY = "analysis_state";

interface SavedState {
  task: TaskType | "";
  model: string;
  explainers: string[];
  detectionCache: DetectionCache;
  hiddenExplainers: string[];
  metricWeights: Record<string, number>;
  jobId: string | null;
}

function loadState(): SavedState | null {
  try { return JSON.parse(sessionStorage.getItem(SS_KEY) ?? "null"); } catch { return null; }
}

function persistState(s: SavedState) {
  try { sessionStorage.setItem(SS_KEY, JSON.stringify(s)); } catch {}
}

export default function Home() {
  const [task, setTask] = useState<TaskType | "">("");
  const [inputData, setInputData] = useState<File | Blob | null>(null);
  const [model, setModel] = useState("");
  const [explainers, setExplainers] = useState<string[]>([]);
  const [rankingMetric] = useState("average");
  const [detectionOpen, setDetectionOpen] = useState(false);
  const [detectionCache, setDetectionCache] = useState<DetectionCache>({
    state: "idle", job: null, selected: [], error: null,
  });
  const [hiddenExplainers, setHiddenExplainers] = useState<string[]>([]);
  const [metricWeights, setMetricWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);
  const { job, loading, error, startJob, attachToJob, reset } = useExplainJob();

  // Restore state on mount (returning from Optimizer)
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    const saved = loadState();
    if (!saved) return;
    setTask(saved.task);
    setModel(saved.model);
    setExplainers(saved.explainers);
    setDetectionCache(saved.detectionCache);
    setHiddenExplainers(saved.hiddenExplainers);
    setMetricWeights(saved.metricWeights);
    if (saved.jobId) attachToJob(saved.jobId);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist state whenever it changes
  useEffect(() => {
    persistState({
      task, model, explainers, detectionCache,
      hiddenExplainers, metricWeights,
      jobId: job?.job_id ?? null,
    });
  }, [task, model, explainers, detectionCache, hiddenExplainers, metricWeights, job?.job_id]);

  const handleWeightChange = (metric: string, value: number) =>
    setMetricWeights((prev) => ({ ...prev, [metric]: value }));

  const handleResetWeights = () => setMetricWeights(DEFAULT_WEIGHTS);

  const toggleHidden = (name: string) =>
    setHiddenExplainers((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );

  const displayNames = useMemo(() => {
    const map: Record<string, string> = {};
    detectionCache.job?.results.forEach((r) => { map[r.name] = r.display_name; });
    return map;
  }, [detectionCache.job]);

  const resetDetectionCache = () =>
    setDetectionCache({ state: "idle", job: null, selected: [], error: null });

  const handleTaskChange = (t: TaskType | "") => {
    if (t === task) return;
    setTask(t as TaskType | "");
    setModel("");
    setExplainers([]);
    setInputData(null);
    resetDetectionCache();
    reset();
  };

  const handleGoFromModal = (selected: string[]) => {
    setExplainers(selected);
    setDetectionOpen(false);
    if (!task || !inputData || !model || selected.length === 0) return;
    if (detectionCache.linkedJobId) {
      attachToJob(detectionCache.linkedJobId);
    } else {
      startJob(task, inputData, model, selected, rankingMetric);
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      <main className="flex-1 overflow-hidden max-w-[1600px] w-full mx-auto px-6 py-6">
        <div className="flex gap-6 h-full items-stretch">
          {/* Left Panel */}
          <div className="w-80 flex-shrink-0 flex flex-col gap-2 overflow-y-auto">
            <TaskSelector selected={task} onSelect={handleTaskChange} disabled={loading} />
            <ModelSelector task={task} selected={model} onSelect={(m) => { setModel(m); setExplainers([]); resetDetectionCache(); reset(); }} disabled={loading} />
            <DataInput task={task} onDataReady={(data) => { setInputData(data); resetDetectionCache(); reset(); }} disabled={loading} />

            <div className="space-y-2">
              <button
                onClick={() => setDetectionOpen(true)}
                disabled={!task || !model || !inputData || loading}
                className="w-full py-1.5 text-sm rounded-lg font-semibold text-white bg-blue-500 border-2 border-blue-500 hover:bg-blue-100 hover:border-blue-500 disabled:border-gray-200 disabled:text-gray-400 disabled:bg-white disabled:cursor-not-allowed transition-all flex items-center justify-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                EXPLAIN
              </button>
              {(!task || !model) && (
                <p className="text-xs text-gray-400 text-center">Run detection to select XAI methods</p>
              )}

              {detectionCache.job?.detected_architectures && detectionCache.job.detected_architectures.length > 0 && (
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg border border-gray-100 bg-gray-50 flex-wrap">
                  <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide">Arch</span>
                  {detectionCache.job.detected_architectures.map((arch) => (
                    <span key={arch} className={`text-[11px] font-medium px-2 py-0.5 rounded-md border ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                      {arch}
                    </span>
                  ))}
                </div>
              )}

              {(job || loading) && explainers.length > 0 && (
                <ControlBox
                  explainers={explainers}
                  displayNames={displayNames}
                  job={job}
                  loading={loading}
                  hiddenExplainers={hiddenExplainers}
                  onToggleHidden={toggleHidden}
                  onSetAllHidden={(hidden) => setHiddenExplainers(hidden ? [...explainers] : [])}
                  task={task as TaskType}
                  metricWeights={metricWeights}
                  onWeightChange={handleWeightChange}
                  onResetWeights={handleResetWeights}
                />
              )}
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>
            )}

            {detectionOpen && task && model && (
              <ExplainerDetectionModal
                task={task as TaskType}
                model={model}
                inputData={inputData}
                cache={detectionCache}
                onCacheChange={setDetectionCache}
                onGo={handleGoFromModal}
                onClose={() => setDetectionOpen(false)}
              />
            )}
          </div>

          {/* Right Panel */}
          <div className="flex-1 min-w-0 flex flex-col gap-4 overflow-hidden">
            {!job && !loading && <WelcomePanel />}

            {job?.predictions && (
              <PredictionInfo
                dataUrl={job.original_data_url}
                predictions={job.predictions}
                task={job.task}
              />
            )}

            {(job || loading) && (
              <ResultsPanel
                results={job?.results ?? []}
                task={job?.task ?? task as any}
                rankingMetric={job?.ranking_metric ?? "average"}
                job={job}
                loading={loading}
                hiddenExplainers={hiddenExplainers}
                metricWeights={metricWeights}
                onWeightChange={handleWeightChange}
                onResetWeights={handleResetWeights}
                className="flex-1 min-h-0"
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
