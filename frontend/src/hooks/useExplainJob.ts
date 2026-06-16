"use client";
import { useState, useRef, useCallback } from "react";
import { JobStatus, TaskType } from "@/lib/types";
import { submitExplainJob, getJobStatus } from "@/lib/api";

const POLL_INTERVAL = 2000;
const MAX_POLL_TIME = 5 * 60 * 1000;

export function useExplainJob() {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startJob = useCallback(
    async (task: TaskType, file: File | Blob, modelName: string, explainerNames: string[], rankingMetric: string) => {
      setLoading(true);
      setError(null);
      setJob(null);
      stopPolling();

      try {
        const jobId = await submitExplainJob(task, file, modelName, explainerNames, rankingMetric);
        startTimeRef.current = Date.now();

        const poll = async () => {
          try {
            const status = await getJobStatus(jobId);
            setJob(status);

            if (status.status === "completed" || status.status === "failed") {
              stopPolling();
              setLoading(false);
              if (status.status === "failed") {
                setError(status.error_message || "Job failed.");
              }
            }

            if (Date.now() - startTimeRef.current > MAX_POLL_TIME) {
              stopPolling();
              setLoading(false);
              setError("Job timed out after 5 minutes.");
            }
          } catch { /* retry silently */ }
        };

        await poll();
        timerRef.current = setInterval(poll, POLL_INTERVAL);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to start job");
        setLoading(false);
      }
    },
    [stopPolling],
  );

  const attachToJob = useCallback(
    async (jobId: string) => {
      setLoading(true);
      setError(null);
      setJob(null);
      stopPolling();
      startTimeRef.current = Date.now();

      // First poll — if job doesn't exist (404), stop immediately
      try {
        const status = await getJobStatus(jobId);
        setJob(status);
        if (status.status === "completed" || status.status === "failed") {
          setLoading(false);
          if (status.status === "failed") setError(status.error_message || "Job failed.");
          return;
        }
      } catch {
        // Job not found (server restarted) — stop silently
        setLoading(false);
        return;
      }

      const poll = async () => {
        try {
          const status = await getJobStatus(jobId);
          setJob(status);
          if (status.status === "completed" || status.status === "failed") {
            stopPolling();
            setLoading(false);
            if (status.status === "failed") setError(status.error_message || "Job failed.");
          }
          if (Date.now() - startTimeRef.current > MAX_POLL_TIME) {
            stopPolling();
            setLoading(false);
            setError("Job timed out after 5 minutes.");
          }
        } catch {
          // Job disappeared — stop polling
          stopPolling();
          setLoading(false);
        }
      };

      timerRef.current = setInterval(poll, POLL_INTERVAL);
    },
    [stopPolling],
  );

  const reset = useCallback(() => {
    stopPolling();
    setJob(null);
    setLoading(false);
    setError(null);
  }, [stopPolling]);

  return { job, loading, error, startJob, attachToJob, reset };
}
