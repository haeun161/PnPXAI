"use client";
import { useEffect, useState } from "react";
import { TaskType } from "@/lib/types";

interface SampleFile {
  name: string;
  path: string;
  compatible?: boolean;
  reason?: string;
  channels?: number;
  col_names?: string[];
}

// Dataset descriptions, sources & links
const SAMPLE_INFO: Record<string, { desc: string; source: string; url?: string; task: string }> = {
  // Image
  "bird.png": { desc: "Bird image for image classification demo.", source: "Sample image", task: "image" },
  "elephant.png": { desc: "Elephant image for image classification demo.", source: "Sample image", task: "image" },
  "flower.png": { desc: "Flower image for image classification demo.", source: "Sample image", task: "image" },
  // Text
  "positive_review.txt": { desc: "Positive sentiment movie review.", source: "Sample text", task: "text" },
  "negative_review.txt": { desc: "Negative sentiment movie review.", source: "Sample text", task: "text" },
  "neutral_review.txt": { desc: "Neutral sentiment movie review.", source: "Sample text", task: "text" },
  // Time-series
  "boiler.csv": {
    desc: "Simulated industrial boiler sensor data for fault detection & classification. 20 sensor channels (steam pressure, temperatures, damper angle, gas consumption, etc.), 200 timesteps. Binary label: normal vs. abnormal blow-down.",
    source: "IEEE DataPort — Simulated Boiler Data for Fault Detection and Classification",
    url: "https://ieee-dataport.org/open-access/simulated-boiler-data-fault-detection-and-classification",
    task: "timeseries",
  },
  "ecg5000.csv": {
    desc: "ECG heartbeat classification (5 classes: normal + 4 abnormal). 1 channel, 140 timesteps. From UCR ECG5000 dataset.",
    source: "UCR Time Series Archive — ECG5000",
    url: "https://www.timeseriesclassification.com/description.php?Dataset=ECG5000",
    task: "timeseries",
  },
};

interface Props {
  task: TaskType | "";
  model?: string;
  onSampleSelect: (file: Blob, preview: string) => void;
  disabled?: boolean;
}

export default function SampleDataSelector({ task, model, onSampleSelect, disabled }: Props) {
  const [samples, setSamples] = useState<SampleFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [infoOpen, setInfoOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!task) { setSamples([]); setSelected(null); setPreviews({}); return; }
    const url = model ? `/api/samples/${task}?model=${model}` : `/api/samples/${task}`;
    fetch(url)
      .then((r) => r.json())
      .then((data: SampleFile[]) => {
        setSamples(data);
        data.forEach(async (s) => {
          try {
            const res = await fetch(`/api/samples/${task}/${s.name}`);
            const blob = await res.blob();
            if (task === "image") {
              setPreviews((prev) => ({ ...prev, [s.name]: URL.createObjectURL(blob) }));
            } else if (task === "text") {
              const text = await blob.text();
              setPreviews((prev) => ({ ...prev, [s.name]: text }));
            } else if (task === "timeseries") {
              const text = await blob.text();
              setPreviews((prev) => ({ ...prev, [s.name]: text }));
            }
          } catch { /* ignore */ }
        });
      })
      .catch(() => setSamples([]));
  }, [task, model]);

  if (!task || samples.length === 0) return null;

  const handleSelect = async (sample: SampleFile) => {
    if (sample.compatible === false) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/samples/${task}/${sample.name}`);
      const blob = await res.blob();

      let preview = sample.name;
      if (task === "image") {
        preview = URL.createObjectURL(blob);
      } else if (task === "text") {
        preview = await blob.text();
      }

      setSelected(sample.name);
      onSampleSelect(blob, preview);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const displayName = (name: string) => name.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ");

  return (
    <div className="space-y-2">
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(samples.length, 3)}, 1fr)` }}>
        {samples.map((s) => {
          const isIncompat = s.compatible === false;
          return (
            <button
              key={s.name}
              onClick={() => handleSelect(s)}
              disabled={disabled || loading || isIncompat}
              className={`rounded-lg border p-2 transition-colors text-left ${
                isIncompat
                  ? "border-gray-100 bg-gray-50 opacity-40 cursor-not-allowed"
                  : selected === s.name
                  ? "border-blue-500 bg-blue-50 ring-1 ring-blue-300"
                  : "border-gray-200 hover:border-blue-400 hover:bg-blue-50/30"
              } ${disabled ? "opacity-50" : ""}`}
              title={isIncompat ? s.reason : undefined}
            >
              {/* Preview */}
              {task === "image" && previews[s.name] && (
                <img src={previews[s.name]} alt={s.name} className="w-full h-16 object-cover rounded mb-1.5" />
              )}
              {task === "text" && previews[s.name] && (
                <div className="w-full h-16 overflow-hidden rounded bg-gray-50 border border-gray-100 p-1.5 mb-1.5 text-[9px] text-gray-500 leading-tight">
                  {previews[s.name].slice(0, 120)}...
                </div>
              )}
              {task === "timeseries" && previews[s.name] && (
                <div className="w-full h-16 rounded bg-gray-50 border border-gray-100 mb-1.5 flex items-end px-1 pb-1 gap-px overflow-hidden">
                  {(() => {
                    const lines = previews[s.name].split("\n").filter((l) => l && !isNaN(parseFloat(l.split(",")[0])));
                    // Use only first column for preview bar chart
                    const vals = lines.map((l) => parseFloat(l.split(",")[0])).filter((v) => !isNaN(v));
                    if (vals.length === 0) return null;
                    let min = vals[0], max = vals[0];
                    for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
                    const range = max - min || 1;
                    const step = Math.max(1, Math.floor(vals.length / 30));
                    const sampled = vals.filter((_, i) => i % step === 0);
                    return sampled.map((v, i) => (
                      <div
                        key={i}
                        className={`rounded-t-sm flex-1 min-w-[2px] ${isIncompat ? "bg-gray-300" : "bg-blue-400"}`}
                        style={{ height: `${((v - min) / range) * 100}%`, minHeight: 2 }}
                      />
                    ));
                  })()}
                </div>
              )}
              {/* Label + info + Download */}
              <div className="flex items-center justify-between mt-1">
                <div className="flex-1 min-w-0 flex items-center gap-1">
                  <p className={`text-[10px] capitalize truncate ${
                    isIncompat ? "text-gray-400" : selected === s.name ? "text-blue-700 font-semibold" : "text-gray-600"
                  }`}>
                    {displayName(s.name)}
                  </p>
                  {SAMPLE_INFO[s.name] && (
                    <div className="relative flex-shrink-0">
                      <span
                        onClick={(e) => { e.stopPropagation(); setInfoOpen(infoOpen === s.name ? null : s.name); }}
                        className="w-3.5 h-3.5 rounded-full bg-gray-200 hover:bg-blue-200 text-gray-500 hover:text-blue-600 flex items-center justify-center cursor-pointer text-[8px] font-bold transition-colors"
                        title="Dataset info"
                      >?</span>
                      {infoOpen === s.name && (
                        <div
                          className="absolute bottom-5 left-0 z-50 w-56 bg-white border border-gray-200 rounded-lg shadow-lg p-2.5 text-left"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <p className="text-[10px] text-gray-700 leading-snug mb-1.5">{SAMPLE_INFO[s.name].desc}</p>
                          <p className="text-[9px] text-gray-500 leading-snug">
                            <span className="font-semibold">Source:</span>{" "}
                            {SAMPLE_INFO[s.name].url ? (
                              <a
                                href={SAMPLE_INFO[s.name].url}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="text-blue-600 hover:text-blue-800 underline"
                              >
                                {SAMPLE_INFO[s.name].source} ↗
                              </a>
                            ) : (
                              <span>{SAMPLE_INFO[s.name].source}</span>
                            )}
                          </p>
                          <button
                            onClick={(e) => { e.stopPropagation(); setInfoOpen(null); }}
                            className="absolute top-1 right-1 text-gray-300 hover:text-gray-500"
                          >
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  {!isIncompat && (
                    <a
                      href={`/api/samples/${task}/${s.name}`}
                      download={s.name}
                      onClick={(e) => e.stopPropagation()}
                      className="text-gray-300 hover:text-gray-600"
                      title="Download"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                    </a>
                  )}
                </div>
              </div>
              {isIncompat && s.reason && (
                <p className="text-[8px] text-red-400 truncate">{s.reason}</p>
              )}
              {s.channels && s.channels > 1 && !isIncompat && (
                <p className="text-[8px] text-gray-400 truncate">{s.channels} channels</p>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
