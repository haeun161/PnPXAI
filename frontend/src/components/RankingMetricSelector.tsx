"use client";
import { TaskType } from "@/lib/types";

interface Props {
  selected: string;
  onSelect: (metric: string) => void;
  task: TaskType | "";
  disabled?: boolean;
}

const ALL_METRICS = [
  { value: "average", label: "Average (All)" },
  { value: "mu_fidelity", label: "Faithfulness" },
  { value: "abpc", label: "Correctness" },
  { value: "sensitivity", label: "Robustness" },
  { value: "complexity", label: "Compactness" },
];

export default function RankingMetricSelector({ selected, onSelect, task, disabled }: Props) {
  // Time-series drops both classification-only metrics; text keeps AbPC ("Correctness").
  const excluded = task === "timeseries" ? ["mu_fidelity", "abpc"]
    : task === "text" ? ["mu_fidelity"]
    : [];
  const metrics = ALL_METRICS.filter((m) => !excluded.includes(m.value));

  return (
    <div className="space-y-1">
      <label className="block text-sm font-semibold text-gray-700">Ranking Metric</label>
      <div className="flex flex-wrap gap-1.5">
        {metrics.map((m) => (
          <button
            key={m.value}
            onClick={() => onSelect(m.value)}
            disabled={disabled}
            className={`py-1 px-2.5 rounded-lg border text-xs font-medium transition-colors ${
              selected === m.value
                ? "border-blue-500 bg-blue-50 text-blue-700"
                : "border-gray-200 text-gray-600 hover:border-gray-300"
            } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}
