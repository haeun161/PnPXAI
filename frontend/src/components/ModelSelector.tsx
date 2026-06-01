"use client";
import { useEffect, useRef, useState } from "react";
import { ModelInfo, TaskType } from "@/lib/types";
import { getModels, validateHfModel } from "@/lib/api";

interface Props {
  task: TaskType | "";
  selected: string;
  onSelect: (name: string) => void;
  disabled?: boolean;
}

type Mode = "preset" | "custom";
type LoadStatus = "idle" | "loading" | "success" | "error";

function parseHfModelId(input: string): string {
  try {
    const url = new URL(input);
    if (url.hostname === "huggingface.co") {
      const parts = url.pathname.split("/").filter(Boolean);
      return parts.slice(0, 2).join("/");
    }
  } catch {
    // not a URL
  }
  return input.trim();
}

export default function ModelSelector({ task, selected, onSelect, disabled }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [mode, setMode] = useState<Mode>("preset");
  const [hfInput, setHfInput] = useState("");
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("idle");
  const [loadError, setLoadError] = useState("");
  const [loadedDisplayName, setLoadedDisplayName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!task) { setModels([]); return; }
    getModels(task).then(setModels).catch(console.error);
  }, [task]);

  // Reset custom state when task changes
  useEffect(() => {
    setMode("preset");
    setHfInput("");
    setLoadStatus("idle");
    setLoadError("");
    setLoadedDisplayName("");
  }, [task]);

  const handleLoad = async () => {
    const modelId = parseHfModelId(hfInput);
    if (!modelId || !task) return;
    setLoadStatus("loading");
    setLoadError("");
    try {
      const result = await validateHfModel(task as TaskType, modelId);
      setLoadedDisplayName(result.display_name);
      setLoadStatus("success");
      onSelect(result.model_id);
    } catch (e: any) {
      setLoadStatus("error");
      setLoadError(e.message || "Failed to load model");
    }
  };

  const handleModeSwitch = (next: Mode) => {
    setMode(next);
    setLoadStatus("idle");
    setLoadError("");
    setHfInput("");
    setLoadedDisplayName("");
    onSelect(""); // clear current selection
  };

  if (!task) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-gray-700">Select Model</label>
        <p className="text-xs text-gray-400">Select a task first</p>
      </div>
    );
  }

  const selectedModelInfo = models.find((m) => m.name === selected);

  return (
    <div>
      <label className="block text-sm font-semibold text-gray-700 mb-2">Select Model</label>
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        {/* Tabs */}
        <div className="flex bg-gray-50 border-b border-gray-200">
          <button
            onClick={() => handleModeSwitch("preset")}
            disabled={disabled}
            className={`flex-1 text-xs py-2 font-medium transition-colors relative ${
              mode === "preset"
                ? "bg-white text-blue-700 border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Sample
          </button>
          <button
            onClick={() => handleModeSwitch("custom")}
            disabled={disabled}
            className={`flex-1 text-xs py-2 font-medium transition-colors relative ${
              mode === "custom"
                ? "bg-white text-blue-700 border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            URL
          </button>
        </div>

        {/* Content */}
        <div className="bg-white p-3">
          {mode === "preset" ? (
            <>
              <select
                value={selected}
                onChange={(e) => onSelect(e.target.value)}
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
              {selectedModelInfo && (
                <p className="text-xs text-gray-500 mt-1.5">{selectedModelInfo.description}</p>
              )}
            </>
          ) : (
            <div className="space-y-2">
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={hfInput}
                  onChange={(e) => {
                    setHfInput(e.target.value);
                    setLoadStatus("idle");
                    setLoadError("");
                  }}
                  onKeyDown={(e) => e.key === "Enter" && handleLoad()}
                  placeholder="Paste a HuggingFace model URL"
                  disabled={disabled || loadStatus === "loading"}
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
                />
                <button
                  onClick={handleLoad}
                  disabled={!hfInput.trim() || disabled || loadStatus === "loading"}
                  className="px-3 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                >
                  {loadStatus === "loading" ? "Loading..." : "Load"}
                </button>
              </div>

              {loadStatus === "loading" && (
                <p className="text-xs text-blue-600 animate-pulse">
                  Downloading model from HuggingFace Hub, this may take a moment...
                </p>
              )}
              {loadStatus === "success" && (
                <p className="text-xs text-green-600">
                  ✓ Loaded: <span className="font-medium">{loadedDisplayName}</span>
                </p>
              )}
              {loadStatus === "error" && (
                <p className="text-xs text-red-600">{loadError}</p>
              )}
              {loadStatus === "idle" && (
                <p className="text-xs text-gray-400">
                  e.g. https://huggingface.co/microsoft/resnet-50
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
