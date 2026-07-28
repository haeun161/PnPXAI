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

  // A bare line pinned to the very top of the viewport -- no message, no percentage --
  // `fixed` takes it out of document flow, so it overlays whatever's on the page
  // (including the nav bar) regardless of where this component is mounted.
  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-[3px] bg-transparent">
      <div className="h-full bg-blue-500 transition-all duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}
