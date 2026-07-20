"use client";
import { useEffect, useRef, useState } from "react";
import { TaskType } from "@/lib/types";
import TaskSelector from "@/components/TaskSelector";
import DataInput from "@/components/DataInput";
import ModelSelector from "@/components/ModelSelector";
import PredictionInfo from "@/components/PredictionInfo";
import ResultsPanel from "@/components/ResultsPanel";
import ControlBox from "@/components/ControlBox";
import { useExplainJob } from "@/hooks/useExplainJob";
import NavBar from "@/components/NavBar";
import Toast from "@/components/Toast";
import { setOptimizerHandoff } from "@/lib/optimizerHandoff";

function WelcomePanel() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center px-8 -mt-[60px]">
      <img src="/pnpxai_logo.png" alt="PnPXAI Logo" className="w-[410px] mb-10 select-none" draggable={false} />
      <h2 className="text-[1.75rem] font-bold text-gray-900 mb-4">AI 모델이 어떤 근거로 예측했는지 확인해보세요</h2>
      <ol className="text-[1.05rem] text-gray-500 leading-relaxed max-w-2xl text-left list-decimal list-inside space-y-1.5 mb-4">
        <li>분석할 데이터 유형(이미지 / 텍스트 / 시계열)을 선택하세요</li>
        <li>분석에 사용할 모델을 선택하세요</li>
        <li>데이터를 업로드하면, 다양한 설명 기법(XAI)이 예측 근거를 시각화해줍니다</li>
      </ol>
      <p className="text-[1.05rem] text-gray-500">왼쪽에서 시작해보세요.</p>
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
  architectures: string[];
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
  const [architectures, setArchitectures] = useState<string[]>([]);
  const [showArchToast, setShowArchToast] = useState(false);
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
    setArchitectures(saved.architectures ?? []);
    setHiddenExplainers(saved.hiddenExplainers);
    setMetricWeights(saved.metricWeights);
    if (saved.jobId) attachToJob(saved.jobId);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps


  // Persist state whenever it changes
  useEffect(() => {
    persistState({
      task, model, explainers, architectures,
      hiddenExplainers, metricWeights,
      jobId: job?.job_id ?? null,
    });
  }, [task, model, explainers, architectures, hiddenExplainers, metricWeights, job?.job_id]);

  // Keep the Optimizer hand-off in sync so clicking "Optimize" on a ResultCard
  // carries over the actual uploaded file + predictions in memory, instead of
  // the Optimizer page having to re-fetch them from the (possibly gone) job.
  useEffect(() => {
    if (!inputData || !job?.original_data_url) return;
    setOptimizerHandoff({
      dataUrl: job.original_data_url,
      file: inputData,
      predictions: job.predictions ?? null,
    });
  }, [inputData, job?.original_data_url, job?.predictions]);

  const handleWeightChange = (metric: string, value: number) =>
    setMetricWeights((prev) => ({ ...prev, [metric]: value }));

  const handleResetWeights = () => setMetricWeights(DEFAULT_WEIGHTS);

  const toggleHidden = (name: string) =>
    setHiddenExplainers((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );

  const resetAll = () => {
    setExplainers([]);
    setArchitectures([]);
    reset();
  };

  const handleTaskChange = (t: TaskType | "") => {
    if (t === task) return;
    setTask(t as TaskType | "");
    setModel("");
    setInputData(null);
    resetAll();
  };

  const handleExplain = async () => {
    if (!task || !model || !inputData) return;
    resetAll();
    // Fetch available explainers and architecture info in parallel
    try {
      const [explainersRes, archRes] = await Promise.all([
        fetch(`/api/explainers?task=${task}&model=${encodeURIComponent(model)}`),
        fetch(`/api/recommend?task=${task}&model=${encodeURIComponent(model)}`),
      ]);
      const explainerList: { name: string }[] = await explainersRes.json();
      const SELECTED = ["IntegratedGradients", "RAP", "LRPUniformEpsilon", "GuidedGradCam"];
      const allNames = explainerList.map((e) => e.name).filter((n) => SELECTED.includes(n));
      setExplainers(allNames);
      startJob(task, inputData, model, allNames, "average");

      const archData = await archRes.json();
      if (archData.detected_architectures?.length) {
        setArchitectures(archData.detected_architectures);
        setShowArchToast(true);
      }
    } catch {
      // fallback: start with empty explainer list (backend decides)
    }
  };

  const busy = loading;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />
      <Toast
        message="Model Architecture Detection Complete"
        show={showArchToast}
        onHide={() => setShowArchToast(false)}
      />

      <main className="flex-1 overflow-hidden max-w-[1600px] w-full mx-auto px-6 py-6">
        <div className="flex gap-6 h-full items-stretch">
          {/* Left Panel */}
          <div className="w-80 flex-shrink-0 flex flex-col gap-2 overflow-y-auto">
            <TaskSelector selected={task} onSelect={handleTaskChange} disabled={busy} />
            <ModelSelector task={task} selected={model} onSelect={(m) => setModel(m)} disabled={busy} />
            <DataInput task={task} onDataReady={(data) => setInputData(data)} disabled={busy} />

            <div className="space-y-2">
              <button
                onClick={handleExplain}
                disabled={!task || !model || !inputData || busy}
                className="w-full py-1.5 text-sm rounded-lg font-semibold text-white bg-blue-500 border-2 border-blue-500 hover:bg-blue-100 hover:border-blue-500 disabled:border-gray-200 disabled:text-gray-400 disabled:bg-white disabled:cursor-not-allowed transition-all flex items-center justify-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                EXPLAIN
              </button>
              {(!task || !model) && (
                <p className="text-xs text-gray-400 text-center">Select task, model, and data to get started</p>
              )}

              {architectures.length > 0 && (
                <div className="rounded-xl border border-blue-100 bg-gradient-to-br from-blue-50 to-indigo-50 p-3">
                  <div className="flex items-center gap-1.5 mb-2">
                    <svg className="w-3.5 h-3.5 text-blue-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2v-4M9 21H5a2 2 0 01-2-2v-4m0 0h18" />
                    </svg>
                    <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wide">Detected Architecture</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {architectures.map((arch) => (
                      <span key={arch} className={`text-[11px] font-semibold px-2.5 py-1 rounded-lg border shadow-sm ${ARCH_COLORS[arch] ?? "bg-gray-100 text-gray-600 border-gray-200"}`}>
                        {arch}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {(job || loading) && (job?.explainer_names ?? explainers).length > 0 && (
                <ControlBox
                  explainers={job?.explainer_names ?? explainers}
                  displayNames={{}}
                  job={job}
                  loading={loading}
                  hiddenExplainers={hiddenExplainers}
                  onToggleHidden={toggleHidden}
                  onSetAllHidden={(hidden) => setHiddenExplainers(hidden ? [...(job?.explainer_names ?? explainers)] : [])}
                  task={task as TaskType}
                  metricWeights={metricWeights}
                  onWeightChange={handleWeightChange}
                  onResetWeights={handleResetWeights}
                />
              )}
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                {error}
              </div>
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
                loading={busy}
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
