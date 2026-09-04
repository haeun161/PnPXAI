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
  // Text — one statement per severity level (1-5) from the ToxiGen annotated split, each
  // aimed at a different racial or ethnic group. None of them contains a slur or a
  // profanity: ToxiGen is built around implicit toxicity, so the interesting question is
  // which ordinary-looking words the attribution picks out.
  // Band names (non-toxic / ambiguous / toxic) are the paper's own, from its footnote 8:
  // "scores <3: 'non-toxic', =3: 'ambiguous', >3: 'toxic'".
  "toxigen_l1_mexican.txt": {
    desc: "Level 1 — non-toxic, clearly benign. Pride in one's own culture, targeting nobody. Useful as the control: the attribution should have nothing strong to point at.",
    source: "ToxiGen (ACL 2022) — annotated split, target group: mexican",
    url: "https://huggingface.co/datasets/toxigen/toxigen-data",
    task: "text",
  },
  "toxigen_l2_chinese.txt": {
    desc: "Level 2 — still non-toxic: rejecting a stereotype rather than making one. Mentions race explicitly, which is why a keyword-driven model tends to overrate it; watch whether the attribution keys on \"race\" alone.",
    source: "ToxiGen (ACL 2022) — annotated split, target group: chinese",
    url: "https://huggingface.co/datasets/toxigen/toxigen-data",
    task: "text",
  },
  "toxigen_l3_latino.txt": {
    desc: "Level 3 — ambiguous, the paper's own name for the midpoint. A policy claim with no insult in it, which annotators split on. Whatever the attribution highlights here is the model's actual notion of borderline harm.",
    source: "ToxiGen (ACL 2022) — annotated split, target group: latino",
    url: "https://huggingface.co/datasets/toxigen/toxigen-data",
    task: "text",
  },
  "toxigen_l4_middle_east.txt": {
    desc: "Level 4 — toxic: a plain civilizational-superiority claim. No slur, no profanity; the harm is carried entirely by the comparison, so the attribution has to find it in the sentence structure.",
    source: "ToxiGen (ACL 2022) — annotated split, target group: middle_east",
    url: "https://huggingface.co/datasets/toxigen/toxigen-data",
    task: "text",
  },
  "toxigen_l5_native_american.txt": {
    desc: "Level 5 — toxic, very offensive or abusive, yet still slur-free: a flat degrading generalisation about a group. The clearest test of whether the attribution lands on the stereotype itself.",
    source: "ToxiGen (ACL 2022) — annotated split, target group: native_american",
    url: "https://huggingface.co/datasets/toxigen/toxigen-data",
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

// The ToxiGen samples are named by level and target group, and the full filename
// ("toxigen l5 native american") needs ~130px against a card label field of ~95px, so it
// was arriving truncated to "Toxigen l...". The level is the one thing a viewer must be
// able to read, so it leads: "L5 · Native American".
const TOXIGEN_SAMPLE = /^toxigen_l(\d+)_(.+)$/;

const displayName = (name: string) => {
  const base = name.replace(/\.[^.]+$/, "");
  const m = TOXIGEN_SAMPLE.exec(base);
  return m ? `L${m[1]} · ${m[2].replace(/_/g, " ")}` : base.replace(/[_-]/g, " ");
};

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
    // Three across is right for the image samples (short names, a thumbnail that reads at
    // any size). Text needs the width: the label field at three columns is ~45px, and the
    // 120-character preview at ~75px is unreadable. Two columns doubles both.
    const maxCols = task === "text" ? 2 : 3;
    const cols = Math.min(samples.length, maxCols);
    return (
      // minmax(0, 1fr) rather than a bare 1fr: the default `auto` floor lets a column grow
      // to its content's min-content width, which is how a grid overflows the box it was
      // told to fill.
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {samples.map((s, idx) => {
          const isIncompat = s.compatible === false;
          const info = SAMPLE_INFO[s.name];
          // The info popover is wider than a card, so anchoring it left would push it out
          // of the panel from the rightmost column. Flip it there.
          const lastCol = idx % cols === cols - 1;
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
                  <p className={`text-[10px] capitalize leading-tight break-words ${
                    isIncompat ? "text-gray-400" : selected === s.name ? "text-blue-700 font-semibold" : "text-gray-600"
                  }`}>
                    {displayName(s.name)}
                  </p>
                  {info && (
                    <div
                      className="relative flex-shrink-0"
                      // Opened by clicking the badge, dismissed by simply moving away — no
                      // second click. mouseleave counts the popover as "inside" because it
                      // is a descendant, absolute positioning notwithstanding, so this only
                      // fires once the pointer has left the badge *and* the panel.
                      onMouseLeave={() => setInfoOpen((cur) => (cur === s.name ? null : cur))}
                    >
                      <span
                        onClick={(e) => { e.stopPropagation(); setInfoOpen(infoOpen === s.name ? null : s.name); }}
                        className="w-3.5 h-3.5 rounded-full bg-gray-200 hover:bg-blue-200 text-gray-500 hover:text-blue-600 flex items-center justify-center cursor-pointer text-[8px] font-bold transition-colors"
                        title="Dataset info"
                      >?</span>
                      {infoOpen === s.name && (
                        // The 6px gap above the badge is padding *on the popover* rather
                        // than an offset, so it is hoverable: crossing it to reach the panel
                        // no longer counts as leaving, which a plain `bottom-5` offset would
                        // have made it (closing the panel before the pointer arrived).
                        <div
                          className={`absolute bottom-full z-50 w-52 pb-1.5 ${
                            lastCol ? "right-0" : "left-0"
                          }`}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-2.5 text-left">
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
