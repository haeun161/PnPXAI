"use client";
import { useState } from "react";
import { TaskType } from "@/lib/types";
import TaskSelector from "@/components/TaskSelector";
import DataInput from "@/components/DataInput";
import ModelSelector from "@/components/ModelSelector";
import ExplainerSelector from "@/components/ExplainerSelector";
import RankingMetricSelector from "@/components/RankingMetricSelector";
import PredictionInfo from "@/components/PredictionInfo";
import ResultsPanel from "@/components/ResultsPanel";
import ProgressIndicator from "@/components/ProgressIndicator";
import { useExplainJob } from "@/hooks/useExplainJob";

export default function Home() {
  const [task, setTask] = useState<TaskType | "">("");
  const [inputData, setInputData] = useState<File | Blob | null>(null);
  const [model, setModel] = useState("");
  const [explainers, setExplainers] = useState<string[]>([]);
  const [rankingMetric, setRankingMetric] = useState("average");
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
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">XAI Demo Platform <span className="text-sm font-normal text-blue-600">v2</span></h1>
            <p className="text-sm text-gray-500">Multi-modal eXplainable AI powered by PnPXAI</p>
          </div>
          <div className="flex items-center gap-2">
            <a href="/optimizer" className="text-sm text-green-600 hover:text-green-700 border border-green-200 rounded-lg px-3 py-1.5">
              Optimizer
            </a>
            {(job || loading) && (
              <button onClick={handleReset} className="text-sm text-gray-500 hover:text-gray-700 border border-gray-300 rounded-lg px-3 py-1.5">
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex gap-6">
          {/* Left Panel */}
          <div className="w-80 flex-shrink-0 space-y-4">
            <TaskSelector selected={task} onSelect={handleTaskChange} disabled={loading} />
            <ModelSelector task={task} selected={model} onSelect={(m) => { setModel(m); setExplainers([]); }} disabled={loading} />
            <DataInput task={task} onDataReady={(data) => setInputData(data)} disabled={loading} />
            <ExplainerSelector task={task} model={model} selected={explainers} onSelect={setExplainers} disabled={loading} />

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
          </div>

          {/* Right Panel */}
          <div className="flex-1 space-y-4">
            <ProgressIndicator job={job} loading={loading} />

            {job?.predictions && (
              <PredictionInfo
                dataUrl={job.original_data_url}
                predictions={job.predictions}
                task={job.task}
              />
            )}

            {job && (
              <ResultsPanel
                results={job.results}
                task={job.task}
                rankingMetric={job.ranking_metric}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
