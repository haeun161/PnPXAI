"use client";
import { useState } from "react";
import { TaskType } from "@/lib/types";
import ImageUploader from "./ImageUploader";
import TextInput from "./TextInput";
import TimeSeriesInput from "./TimeSeriesInput";
import SampleDataSelector from "./SampleDataSelector";

interface Props {
  task: TaskType | "";
  model?: string;
  onDataReady: (data: File | Blob, preview: string) => void;
  disabled?: boolean;
}

export default function DataInput({ task, model, onDataReady, disabled }: Props) {
  const [mode, setMode] = useState<"sample" | "upload">("sample");

  if (!task) return null;

  return (
    <div>
      <label className="block text-sm font-semibold text-gray-700 mb-1">Upload Data</label>
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        {/* Tabs */}
        <div className="flex bg-gray-50 border-b border-gray-200">
          <button
            onClick={() => setMode("sample")}
            disabled={disabled}
            className={`flex-1 text-xs py-2 font-medium transition-colors relative ${
              mode === "sample"
                ? "bg-white text-blue-700 border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Sample Data
          </button>
          <button
            onClick={() => setMode("upload")}
            disabled={disabled}
            className={`flex-1 text-xs py-2 font-medium transition-colors relative ${
              mode === "upload"
                ? "bg-white text-blue-700 border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Upload
          </button>
        </div>

        {/* Content */}
        <div className="bg-white p-3 max-h-36 overflow-y-auto">
          {mode === "sample" && (
            <SampleDataSelector task={task} model={model} onSampleSelect={onDataReady} disabled={disabled} />
          )}

          {mode === "upload" && (
            <>
              {task === "image" && (
                <ImageUploader onImageSelect={(file) => onDataReady(file, file.name)} disabled={disabled} />
              )}
              {task === "text" && (
                <TextInput onTextReady={onDataReady} disabled={disabled} />
              )}
              {task === "timeseries" && (
                <TimeSeriesInput onDataReady={onDataReady} disabled={disabled} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
