"use client";
import { useEffect, useState } from "react";
import { SampleFile, TaskType } from "@/lib/types";
import { getSamples } from "@/lib/api";

// Dataset descriptions, sources & links
const SAMPLE_INFO: Record<string, { desc: string; source: string; url?: string; task: string }> = {
  // Image
  "bird.png": { desc: "Bird image for image classification demo.", source: "Sample image", task: "image" },
  "elephant.png": { desc: "Elephant image for image classification demo.", source: "Sample image", task: "image" },
  "flower.png": { desc: "Flower image for image classification demo.", source: "Sample image", task: "image" },
  // Text — one post per gold label from the HateXplain test split
  "hate_speech_post.txt": {
    desc: "Hate speech: a racial slur set against \"white man\", so the post attacks a group rather than a person. Annotators marked the slur itself as the reason.",
    source: "HateXplain (AAAI 2021) — test split",
    url: "https://github.com/hate-alert/HateXplain",
    task: "text",
  },
  "normal_post.txt": {
    desc: "Normal: harmless everyday post, with no insult and no group targeted.",
    source: "HateXplain (AAAI 2021) — test split",
    url: "https://github.com/hate-alert/HateXplain",
    task: "text",
  },
  "offensive_post.txt": {
    desc: "Offensive, not hate speech: an insult aimed at one person, targeting no group. The slur appears twice, so the attribution should light up in both places.",
    source: "HateXplain (AAAI 2021) — test split",
    url: "https://github.com/hate-alert/HateXplain",
    task: "text",
  },
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
  onSampleSelect: (file: Blob, preview: string, name: string) => void;
  disabled?: boolean;
}

const displayName = (name: string) => name.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ");

export default function SampleDataSelector({ task, model, onSampleSelect, disabled }: Props) {
  const [samples, setSamples] = useState<SampleFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState("");
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [infoOpen, setInfoOpen] = useState<string | null>(null);

  // Images and text are small enough to preview as thumbnails. Time-series samples run
  // to tens of MB, so that task gets a dropdown and nothing is fetched until selection.
  const usePreviewCards = task === "image" || task === "text";

  useEffect(() => {
    if (!task) { setSamples([]); setSelected(""); return; }
    getSamples(task as TaskType, model).then(setSamples).catch(() => setSamples([]));
    setSelected("");
  }, [task, model]);

  useEffect(() => {
    if (!usePreviewCards) { setPreviews({}); return; }
    let cancelled = false;
    samples.forEach(async (s) => {
      try {
        const res = await fetch(`/api/samples/${task}/${s.name}`);
        const blob = await res.blob();
        const value = task === "image" ? URL.createObjectURL(blob) : await blob.text();
        if (!cancelled) setPreviews((prev) => ({ ...prev, [s.name]: value }));
      } catch { /* ignore */ }
    });
    return () => { cancelled = true; };
  }, [samples, usePreviewCards, task]);

  if (!task || samples.length === 0) return null;

  const handleSelect = async (name: string) => {
    setSelected(name);
    if (!name) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/samples/${task}/${name}`);
      const blob = await res.blob();
      const preview = task === "image" ? URL.createObjectURL(blob)
        : task === "text" ? await blob.text()
        : name;
      onSampleSelect(blob, preview, name);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  if (usePreviewCards) {
    return (
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(samples.length, 3)}, 1fr)` }}>
        {samples.map((s) => {
          const isIncompat = s.compatible === false;
          const info = SAMPLE_INFO[s.name];
          return (
            <button
              key={s.name}
              onClick={() => handleSelect(s.name)}
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
              {task === "image" && previews[s.name] && (
                <img src={previews[s.name]} alt={s.name} className="w-full h-16 object-contain rounded mb-1.5" />
              )}
              {task === "text" && previews[s.name] && (
                <div className="w-full h-16 overflow-hidden rounded bg-gray-50 border border-gray-100 p-1.5 mb-1.5 text-[9px] text-gray-500 leading-tight">
                  {previews[s.name].slice(0, 120)}...
                </div>
              )}

              <div className="flex items-center justify-between mt-1">
                <div className="flex-1 min-w-0 flex items-center gap-1">
                  <p className={`text-[10px] capitalize truncate ${
                    isIncompat ? "text-gray-400" : selected === s.name ? "text-blue-700 font-semibold" : "text-gray-600"
                  }`}>
                    {displayName(s.name)}
                  </p>
                  {info && (
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
                          <p className="text-[10px] text-gray-700 leading-snug mb-1.5">{info.desc}</p>
                          <p className="text-[9px] text-gray-500 leading-snug">
                            <span className="font-semibold">Source:</span>{" "}
                            {info.url ? (
                              <a href={info.url} target="_blank" rel="noopener noreferrer"
                                 onClick={(e) => e.stopPropagation()}
                                 className="text-blue-600 hover:text-blue-800 underline">
                                {info.source} ↗
                              </a>
                            ) : (
                              <span>{info.source}</span>
                            )}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <a
                  href={`/api/samples/${task}/${s.name}`}
                  download={s.name}
                  onClick={(e) => e.stopPropagation()}
                  className="text-gray-300 hover:text-gray-600 flex-shrink-0"
                  title="Download"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </a>
              </div>
              {isIncompat && s.reason && <p className="text-[8px] text-red-400 truncate">{s.reason}</p>}
            </button>
          );
        })}
      </div>
    );
  }

  const info = selected ? SAMPLE_INFO[selected] : undefined;
  return (
    <div className="space-y-2">
      <select
        value={selected}
        onChange={(e) => handleSelect(e.target.value)}
        disabled={disabled || loading}
        className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
      >
        <option value="">Choose data...</option>
        {samples.map((s) => (
          <option key={s.name} value={s.name} disabled={s.compatible === false}>
            {displayName(s.name)}
            {s.channels && s.channels > 1 ? ` (${s.channels} channels)` : ""}
            {s.compatible === false ? ` — ${s.reason ?? "incompatible"}` : ""}
          </option>
        ))}
      </select>

      {loading && <p className="text-xs text-blue-600 animate-pulse">Loading data...</p>}

      {info && (
        <p className="text-[10px] text-gray-400 leading-snug">
          {info.desc}{" "}
          {info.url ? (
            <a href={info.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700 underline">
              {info.source} ↗
            </a>
          ) : (
            info.source
          )}
        </p>
      )}

      {selected && (
        <a
          href={`/api/samples/${task}/${selected}`}
          download={selected}
          className="inline-flex items-center gap-1 text-[10px] text-gray-400 hover:text-gray-600"
        >
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download
        </a>
      )}
    </div>
  );
}
