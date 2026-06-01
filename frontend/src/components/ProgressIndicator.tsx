"use client";
import { JobStatus } from "@/lib/types";

interface Props {
  job: JobStatus | null;
  loading: boolean;
}

export default function ProgressIndicator({ job, loading }: Props) {
  if (!loading && !job) return null;

  const total = job?.explainer_names.length ?? 0;
  const done = job?.results.filter(
    (r) => r.status === "completed" || r.status === "not_supported" || r.status === "failed"
  ).length ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-blue-700 font-medium">
          {job?.status === "completed" ? "Analysis Complete" : "Analyzing..."}
        </span>
        <span className="text-blue-500">{Math.round(pct)}%</span>
      </div>
      <div className="h-2 rounded-full bg-blue-100 overflow-hidden">
        <div className="h-full rounded-full bg-blue-500 transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
