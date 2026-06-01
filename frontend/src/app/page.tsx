"use client";
import { useState } from "react";
import { TaskType } from "@/lib/types";
import TaskSelector from "@/components/TaskSelector";
import DataInput from "@/components/DataInput";
import ModelSelector from "@/components/ModelSelector";
import ExplainerDetectionModal from "@/components/ExplainerDetectionModal";
import PredictionInfo from "@/components/PredictionInfo";
import ResultsPanel from "@/components/ResultsPanel";
import { useExplainJob } from "@/hooks/useExplainJob";
import NavBar from "@/components/NavBar";

export default function Home() {
  const [task, setTask] = useState<TaskType | "">("");
  const [inputData, setInputData] = useState<File | Blob | null>(null);
  const [model, setModel] = useState("");
  const [explainers, setExplainers] = useState<string[]>([]);
  const [rankingMetric, setRankingMetric] = useState("average");
  const [detectionOpen, setDetectionOpen] = useState(false);
  const { job, loading, error, startJob, reset } = useExplainJob();

  const canRun = task && inputData && model && explainers.length > 0 && !loading;

  const handleTaskChange = (t: TaskType) => {
    setTask(t);
    setModel("");
    setExplainers([]);
    setInputData(null);
  };

  const handleRun = () => {
    if (!task || !inputData || !model || explainers.length === 0) return;
    startJob(task, inputData, model, explainers, rankingMetric);
  };

  const handleReset = () => {
    reset();
    setTask("");
    setInputData(null);
    setModel("");
    setExplainers([]);
    setRankingMetric("mu_fidelity");
  };

  return (
    <div className="min-h-screen">
      <NavBar />

      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex gap-6">
          {/* Left Panel */}
          <div className="w-80 flex-shrink-0 space-y-4">
            <TaskSelector selected={task} onSelect={handleTaskChange} disabled={loading} />
            <ModelSelector task={task} selected={model} onSelect={(m) => { setModel(m); setExplainers([]); }} disabled={loading} />
            <DataInput task={task} onDataReady={(data) => setInputData(data)} disabled={loading} />

            {/* Explainer Detection */}
            <div className="space-y-2">
              <button
                onClick={() => setDetectionOpen(true)}
                disabled={!task || !model || loading}
                className="w-full py-2.5 rounded-lg font-semibold border-2 border-blue-600 text-blue-600 bg-white hover:bg-blue-50 disabled:border-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Explainer Detection
              </button>
              {explainers.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {explainers.slice(0, 4).map((name) => (
                    <span key={name} className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 rounded px-1.5 py-0.5">
                      {name}
                    </span>
                  ))}
                  {explainers.length > 4 && (
                    <span className="text-[10px] text-gray-400">+{explainers.length - 4} more</span>
                  )}
                </div>
              ) : (task && model) ? (
                <p className="text-xs text-gray-400 text-center">Run detection to select XAI methods</p>
              ) : null}
            </div>

            <button
              onClick={handleRun}
              disabled={!canRun}
              className="w-full py-2.5 rounded-lg font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>
            )}

            {detectionOpen && task && model && (
              <ExplainerDetectionModal
                task={task as TaskType}
                model={model}
                inputData={inputData}
                onSave={setExplainers}
                onClose={() => setDetectionOpen(false)}
              />
            )}
          </div>

          {/* Right Panel */}
          <div className="flex-1 space-y-4">
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
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
