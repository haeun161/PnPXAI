"use client";
import { useEffect, useState } from "react";
import { ModelInfo, TaskType } from "@/lib/types";
import { getModels } from "@/lib/api";

interface Props {
  task: TaskType | "";
  selected: string;
  onSelect: (name: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({ task, selected, onSelect, disabled }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    if (!task) { setModels([]); return; }
    getModels(task).then(setModels).catch(console.error);
  }, [task]);

  if (!task) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-gray-700">Select Model</label>
        <p className="text-xs text-gray-400">Select a task first</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-gray-700">Select Model</label>
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
      {selected && models.find((m) => m.name === selected) && (
        <p className="text-xs text-gray-500">
          {models.find((m) => m.name === selected)!.description}
        </p>
      )}
    </div>
  );
}
