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
  // Bumped on every startJob/attachToJob/reset call. Any in-flight poll from an
  // older call checks its captured value against this ref before touching state
  // or installing an interval — so a stale call can never clobber a newer one,
  // even if their async steps (submit, first poll) resolve out of order.
  const generationRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startJob = useCallback(
    async (task: TaskType, file: File | Blob, modelName: string, explainerNames: string[], rankingMetric: string) => {
      const myGeneration = ++generationRef.current;
      setLoading(true);
      setError(null);
      setJob(null);
      stopPolling();

      try {
        const jobId = await submitExplainJob(task, file, modelName, explainerNames, rankingMetric);
        if (myGeneration !== generationRef.current) return;
        startTimeRef.current = Date.now();

        const poll = async () => {
          if (myGeneration !== generationRef.current) return;
          try {
            const status = await getJobStatus(jobId);
            if (myGeneration !== generationRef.current) return;
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
        if (myGeneration !== generationRef.current) return;
        timerRef.current = setInterval(poll, POLL_INTERVAL);
      } catch (e) {
        if (myGeneration !== generationRef.current) return;
        setError(e instanceof Error ? e.message : "Failed to start job");
        setLoading(false);
      }
    },
    [stopPolling],
  );

  const attachToJob = useCallback(
    async (jobId: string) => {
      const myGeneration = ++generationRef.current;
      setLoading(true);
      setError(null);
      setJob(null);
      stopPolling();
      startTimeRef.current = Date.now();

      // First poll — if job doesn't exist (404), stop immediately
      try {
        const status = await getJobStatus(jobId);
        if (myGeneration !== generationRef.current) return;
        setJob(status);
        if (status.status === "completed" || status.status === "failed") {
          setLoading(false);
          if (status.status === "failed") setError(status.error_message || "Job failed.");
          return;
        }
      } catch {
        if (myGeneration !== generationRef.current) return;
        // Job not found (server restarted) — stop silently
        setLoading(false);
        return;
      }

      const poll = async () => {
        if (myGeneration !== generationRef.current) return;
        try {
          const status = await getJobStatus(jobId);
          if (myGeneration !== generationRef.current) return;
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
          if (myGeneration !== generationRef.current) return;
          // Job disappeared — stop polling
          stopPolling();
          setLoading(false);
        }
      };

      if (myGeneration !== generationRef.current) return;
      timerRef.current = setInterval(poll, POLL_INTERVAL);
    },
    [stopPolling],
  );

  const reset = useCallback(() => {
    generationRef.current++;
    stopPolling();
    setJob(null);
    setLoading(false);
    setError(null);
  }, [stopPolling]);

  return { job, loading, error, startJob, attachToJob, reset };
}
