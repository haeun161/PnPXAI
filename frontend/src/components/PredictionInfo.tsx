"use client";
import { PredictionItem, TaskType } from "@/lib/types";
import { useEffect, useState } from "react";

interface Props {
  dataUrl: string | null;
  predictions: PredictionItem[] | null;
  task: TaskType;
}

export default function PredictionInfo({ dataUrl, predictions, task }: Props) {
  const [textContent, setTextContent] = useState<string | null>(null);

  useEffect(() => {
    if (task === "text" && dataUrl) {
      fetch(dataUrl).then((r) => r.text()).then(setTextContent).catch(() => setTextContent(null));
    }
  }, [task, dataUrl]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-4 py-3">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Input & Prediction</h3>
      <div className="flex gap-4">
        {/* Original data display */}
        {task === "image" && dataUrl && (
          <img src={dataUrl} alt="Original" className="max-h-40 rounded-lg object-contain border" />
        )}
        {task === "text" && (
          <div className="max-h-40 w-56 overflow-y-auto rounded-lg border bg-gray-50 p-2.5 text-sm text-gray-800 flex-shrink-0">
            {textContent || "Loading..."}
          </div>
        )}
        {task === "timeseries" && (
          <div className="w-56 h-24 rounded-lg border bg-gray-50 flex items-center justify-center text-xs text-gray-400 flex-shrink-0">
            Time-series data loaded
          </div>
        )}

        {/* Predictions */}
        {predictions && predictions.length > 0 && (
          <div className="flex-1 space-y-1.5">
            <p className="text-xs font-medium text-gray-500 uppercase">Top-3 Predictions</p>
            {predictions.slice(0, 3).map((pred, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex-1">
                  <div className="flex justify-between text-sm">
                    <span className={i === 0 ? "font-semibold text-blue-700" : "text-gray-700"}>
                      {pred.class_name}
                    </span>
                    <span className="text-gray-500">{pred.probability.toFixed(1)}%</span>
                  </div>
                  <div className="mt-0.5 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${i === 0 ? "bg-blue-500" : "bg-gray-300"}`}
                      style={{ width: `${Math.min(pred.probability, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
