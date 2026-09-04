"use client";
import { useEffect, useRef, useState } from "react";
import { ModelInfo, TaskType } from "@/lib/types";
import { getModels, uploadModel, validateHfModel } from "@/lib/api";
import ImageUploader from "./ImageUploader";
import TextInput from "./TextInput";
import TimeSeriesInput from "./TimeSeriesInput";
import SampleDataSelector from "./SampleDataSelector";

interface Props {
  task: TaskType | "";
  model: string;
  onModelSelect: (name: string) => void;
  // `dataName` is the sample file's name, or undefined when the user uploaded their own.
  onDataReady: (data: Blob, preview: string, dataName?: string) => void;
  disabled?: boolean;
}

type Mode = "sample" | "custom";
type Status = "idle" | "loading" | "success" | "error";

// For image/text the URL box takes a HuggingFace model page. Time series accepts that too
// (a *ForPrediction repo is loaded with transformers) and additionally a direct link to a
// checkpoint file, which the server downloads and treats like an upload.
const URL_FIELD: Record<string, { placeholder: string; hint: string }> = {
  image: {
    placeholder: "Input HuggingFace model URL",
    hint: "ex. https://huggingface.co/microsoft/resnet-50",
  },
  text: {
    placeholder: "Input HuggingFace model URL",
    hint: "ex. https://huggingface.co/tomh/toxigen_roberta",
  },
  timeseries: {
    placeholder: "Input HuggingFace or checkpoint URL",
    hint: "ex. https://huggingface.co/ibm-granite/granite-timeseries-patchtst — or link a .pth/.pt file directly",
  },
};

function parseHfModelId(input: string): string {
  try {
    const url = new URL(input);
    if (url.hostname === "huggingface.co") {
      return url.pathname.split("/").filter(Boolean).slice(0, 2).join("/");
    }
  } catch {
    // not a URL
  }
  return input.trim();
}

export default function ModelDataSelector({ task, model, onModelSelect, onDataReady, disabled }: Props) {
  const [mode, setMode] = useState<Mode>("sample");
  const [models, setModels] = useState<ModelInfo[]>([]);

  const [modelStatus, setModelStatus] = useState<Status>("idle");
  const [modelError, setModelError] = useState("");
  const [customModelName, setCustomModelName] = useState("");
  const [hfInput, setHfInput] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!task) { setModels([]); return; }
    getModels(task).then(setModels).catch(console.error);
  }, [task]);

  // Models and data are both task-scoped, so a task change resets the whole box.
  useEffect(() => {
    setMode("sample");
    setModelStatus("idle");
    setModelError("");
    setCustomModelName("");
    setHfInput("");
  }, [task]);

  if (!task) {
    return (
      <div className="space-y-2">
        <label className="block text-base font-semibold text-gray-700">Model &amp; Data</label>
        <p className="text-xs text-gray-400">Select a task first</p>
      </div>
    );
  }

  const reset = () => {
    onModelSelect("");
    setModelStatus("idle");
    setModelError("");
    setCustomModelName("");
    setHfInput("");
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    reset();
  };

  const handleWeightsUpload = async (file: File) => {
    setModelStatus("loading");
    setModelError("");
    try {
      const result = await uploadModel(task as TaskType, file);
      onModelSelect(result.model_id);
      setCustomModelName(result.display_name);
      setModelStatus("success");
    } catch (e: any) {
      setModelStatus("error");
      setModelError(e.message || "Failed to load model");
    }
  };

  const handleHfLoad = async () => {
    // Time-series URLs point at a checkpoint file, not a Hub model page — pass them
    // through untouched so the server can download the file itself.
    const id = task === "timeseries" ? hfInput.trim() : parseHfModelId(hfInput);
    if (!id) return;
    setModelStatus("loading");
    setModelError("");
    try {
      const result = await validateHfModel(task as TaskType, id);
      onModelSelect(result.model_id);
      setCustomModelName(result.display_name);
      setModelStatus("success");
    } catch (e: any) {
      setModelStatus("error");
      setModelError(e.message || "Failed to load model");
    }
  };

  const urlField = URL_FIELD[task];

  const tabClass = (active: boolean) =>
    `flex-1 text-xs py-2 font-medium transition-colors ${
      active ? "bg-white text-blue-700 border-b-2 border-blue-500" : "text-gray-500 hover:text-gray-700"
    }`;

  return (
    <div>
      <label className="block text-base font-semibold text-gray-700 mb-1">Model &amp; Data</label>
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        <div className="flex bg-gray-50 border-b border-gray-200">
          <button onClick={() => switchMode("sample")} disabled={disabled} className={tabClass(mode === "sample")}>
            Sample
          </button>
          <button onClick={() => switchMode("custom")} disabled={disabled} className={tabClass(mode === "custom")}>
            Custom
          </button>
        </div>

        <div className="bg-white p-3 space-y-3">
          {mode === "sample" ? (
            <>
              <div className="space-y-1">
                <p className="text-[11px] font-medium text-gray-500 uppercase">Model</p>
                <select
                  value={model}
                  onChange={(e) => onModelSelect(e.target.value)}
                  disabled={disabled}
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                >
                  <option value="">Choose a model...</option>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.architecture ? `${m.display_name} (${m.architecture})` : m.display_name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <p className="text-[11px] font-medium text-gray-500 uppercase">Data</p>
                {model ? (
                  // The list is already narrowed server-side to what this model was
                  // trained on. Keyed by model so switching models clears the selection.
                  <SampleDataSelector
                    key={model}
                    task={task}
                    model={model}
                    onSampleSelect={(blob, preview, name) => onDataReady(blob, preview, name)}
                    disabled={disabled}
                  />
                ) : (
                  <p className="text-xs text-gray-400">Select a model first.</p>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <p className="text-[11px] font-medium text-gray-500 uppercase shrink-0">Model</p>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".pth,.pt,.bin,.safetensors"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleWeightsUpload(f); }}
                    disabled={disabled || modelStatus === "loading"}
                    className="flex-1 min-w-0 text-xs text-gray-600 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 file:text-xs file:font-medium hover:file:bg-blue-100 disabled:opacity-50"
                  />
                </div>
                {urlField && (
                  <div className="flex gap-2 pt-1">
                    <input
                      type="text"
                      value={hfInput}
                      onChange={(e) => setHfInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleHfLoad()}
                      placeholder={urlField.placeholder}
                      disabled={disabled || modelStatus === "loading"}
                      className="flex-1 min-w-0 rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                    />
                    <button
                      onClick={handleHfLoad}
                      disabled={!hfInput.trim() || disabled || modelStatus === "loading"}
                      className="px-2.5 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                      Load
                    </button>
                  </div>
                )}
                {modelStatus === "loading" && <p className="text-xs text-blue-600 animate-pulse">Loading model...</p>}
                {modelStatus === "success" && (
                  <p className="text-xs text-green-600">✓ Loaded: <span className="font-medium">{customModelName}</span></p>
                )}
                {modelStatus === "error" && <p className="text-xs text-red-600">{modelError}</p>}
                {modelStatus === "idle" && (
                  <p className="text-[10px] text-gray-400 break-all">
                    {urlField
                      ? urlField.hint
                      : "Checkpoint saved as {'state_dict': ..., 'config': ...}"}
                  </p>
                )}
              </div>

              <div className="space-y-1">
                <p className="text-[11px] font-medium text-gray-500 uppercase">Data</p>
                {modelStatus !== "success" ? (
                  <p className="text-xs text-gray-400">Load a model first.</p>
                ) : (
                  <>
                    {task === "image" && (
                      <ImageUploader onImageSelect={(file) => onDataReady(file, file.name)} disabled={disabled} />
                    )}
                    {task === "text" && <TextInput onTextReady={onDataReady} disabled={disabled} />}
                    {task === "timeseries" && <TimeSeriesInput onDataReady={onDataReady} disabled={disabled} />}
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
