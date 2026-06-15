"use client";
import { TaskType } from "@/lib/types";

interface Props {
  selected: TaskType | "";
  onSelect: (task: TaskType | "") => void;
  disabled?: boolean;
}

const TASKS: { name: TaskType; label: string; icon: string; desc: string; comingSoon?: boolean }[] = [
  { name: "image", label: "Image", icon: "M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z", desc: "Image Classification" },
  { name: "text", label: "Text", icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z", desc: "Text Classification" },
  { name: "timeseries", label: "Time-Series", icon: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6", desc: "Time-Series Classification" },
];

export default function TaskSelector({ selected, onSelect, disabled }: Props) {
  return (
    <div>
      <label className="block text-sm font-semibold text-gray-700 mb-1">Select Task</label>
      <div className="grid grid-cols-3 gap-2">
        {TASKS.map((t) => (
          <button
            key={t.name}
            onClick={() => !t.comingSoon && onSelect(selected === t.name ? "" : t.name)}
            disabled={disabled || t.comingSoon}
            className={`flex flex-col items-center gap-1 rounded-lg border p-3 text-xs transition-colors ${
              t.comingSoon
                ? "border-gray-100 bg-gray-50 text-gray-300 cursor-not-allowed"
                : selected === t.name
                ? "border-blue-500 bg-blue-50 text-blue-700"
                : "border-gray-200 hover:border-gray-300 text-gray-600 cursor-pointer"
            } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={t.icon} />
            </svg>
            <span className="font-medium">{t.label}</span>
            {t.comingSoon && <span className="text-[9px] text-gray-400">Coming Soon</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
