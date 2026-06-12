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
    <div className="flex items-center gap-3 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg">
      <span className="text-xs text-blue-700 font-medium flex-shrink-0">
        {job?.status === "completed" ? "Analysis Complete" : "Analyzing..."}
      </span>
      <div className="flex-1 h-1.5 rounded-full bg-blue-100 overflow-hidden">
        <div className="h-full rounded-full bg-blue-500 transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-blue-500 tabular-nums flex-shrink-0">{Math.round(pct)}%</span>
    </div>
  );
}
