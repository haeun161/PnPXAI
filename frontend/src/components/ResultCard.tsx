"use client";
import { useState, useCallback, useRef } from "react";
import { ExplainerResult, TaskType } from "@/lib/types";

interface Props {
  result: ExplainerResult;
  task: TaskType;
  activeMetrics: string[];
  modelName?: string;
  dataUrl?: string | null;
}

const BAR_HEIGHTS = [45, 85, 35, 100, 60, 75, 40, 90, 55, 70];

export default function ResultCard({ result, task, activeMetrics, modelName, dataUrl }: Props) {
  const [showZoom, setShowZoom] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);

  const zoomUrl = (task === "timeseries" && result.visualization_url)
    ? result.visualization_url.replace(".png", "_expanded.png")
    : result.visualization_url;

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.min(8, Math.max(0.5, z - e.deltaY * 0.001)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    setPan({
      x: dragRef.current.panX + (e.clientX - dragRef.current.startX),
      y: dragRef.current.panY + (e.clientY - dragRef.current.startY),
    });
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const isCompleted = result.status === "completed";
  const isNotSupported = result.status === "not_supported";
  const isFailed = result.status === "failed";
  const isRunning = result.status === "running" || result.status === "pending";

  const faithfulness = (task === "text" || task === "timeseries")
    ? result.abpc
    : (result.mu_fidelity != null && result.abpc != null)
      ? (result.mu_fidelity + result.abpc) / 2
      : result.mu_fidelity ?? result.abpc;

  const metrics = [
    { key: "faithfulness", label: "Faithfulness", value: faithfulness },
    { key: "sensitivity",  label: "Robustness",   value: result.sensitivity },
    { key: "complexity",   label: "Compactness",  value: result.complexity },
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col h-full">
      {/* Zoom modal */}
      {showZoom && zoomUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center overflow-hidden"
          onClick={() => { setShowZoom(false); setZoom(1); setPan({ x: 0, y: 0 }); }}
          onWheel={handleWheel}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
        >
          <button
            onClick={(e) => { e.stopPropagation(); setShowZoom(false); setZoom(1); setPan({ x: 0, y: 0 }); }}
            className="absolute top-4 right-4 text-white bg-black/40 hover:bg-black/60 rounded-full w-8 h-8 flex items-center justify-center transition-colors z-10"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <img
            src={zoomUrl}
            alt={result.display_name}
            onMouseDown={handleMouseDown}
            onClick={(e) => e.stopPropagation()}
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center",
              transition: dragRef.current ? "none" : "transform 0.1s",
              maxWidth: "90vw",
              maxHeight: "90vh",
              objectFit: "contain",
              userSelect: "none",
            }}
          />
        </div>
      )}

      <div className={`relative bg-gray-50 flex-shrink-0 ${task === "text" ? "h-[380px]" : task === "timeseries" ? "flex-1 min-h-0" : "flex-1 min-h-0"}`}>
        {result.rank != null && (
          <div className="absolute top-2 left-2 z-10 w-7 h-7 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center shadow">
            #{result.rank}
          </div>
        )}
        {isCompleted && result.visualization_url && (
          <>
            <img
              src={result.visualization_url}
              alt={result.display_name}
              className="w-full h-full object-contain cursor-zoom-in"
              onDoubleClick={() => { setShowZoom(true); setZoom(1); }}
            />
            <div className="absolute bottom-2 right-2 flex items-center gap-1">
              {task === "timeseries" && (
                <button
                  onClick={() => { setShowZoom(true); setZoom(1); }}
                  className="bg-white/90 hover:bg-white rounded-md p-1 text-gray-600 hover:text-blue-700 shadow-sm transition-colors"
                  title="Show all variables"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
                  </svg>
                </button>
              )}
              {task === "timeseries" && result.visualization_url ? (
                <a
                  href={result.visualization_url.replace(".png", "_bundle.zip")}
                  download={`${result.explainer_name}_attribution.zip`}
                  className="bg-white/90 hover:bg-white rounded-md p-1 text-gray-600 hover:text-blue-700 shadow-sm transition-colors"
                  title="Download (ZIP: all variable graphs + Excel data)"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </a>
              ) : (
                <a
                  href={result.visualization_url ?? ""}
                  download={`${result.explainer_name}_xai_result.png`}
                  className="bg-white/90 hover:bg-white rounded-md p-1 text-gray-600 hover:text-blue-700 shadow-sm transition-colors"
                  title="Download"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </a>
              )}
            </div>
          </>
        )}
        {isCompleted && task === "text" && result.token_attributions && (
          <div className="absolute bottom-0 left-0 right-0 bg-white/90 p-2 max-h-[50%] overflow-y-auto">
            <div className="flex flex-wrap gap-0.5">
              {result.token_attributions.map((ta, i) => (
                <span
                  key={i}
                  className="px-0.5 py-px rounded text-[10px]"
                  style={{
                    backgroundColor: `rgba(239, 68, 68, ${Math.min(ta.score, 1) * 0.7})`,
                    color: ta.score > 0.5 ? "white" : "inherit",
                  }}
                  title={`${ta.token}: ${ta.score.toFixed(3)}`}
                >
                  {ta.token}
                </span>
              ))}
            </div>
          </div>
        )}
        {isNotSupported && (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 p-4">
            <svg className="w-8 h-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <span className="text-xs text-center font-medium text-red-500">Not Supported</span>
          </div>
        )}
        {isFailed && (
          <div className="w-full h-full flex flex-col items-center justify-center text-red-400 p-4">
            <svg className="w-8 h-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-xs text-center">Error</span>
            {result.error_message && (
              <span className="text-xs text-center text-gray-400 mt-1 line-clamp-2">{result.error_message}</span>
            )}
          </div>
        )}
        {isRunning && (
          <div className="w-full h-full flex flex-col items-center justify-center gap-3 bg-gradient-to-b from-gray-50 to-blue-50">
            {/* Triple concentric spinning rings */}
            <div className="relative w-16 h-16 flex items-center justify-center">
              <div
                className="absolute inset-0 rounded-full border-2 border-transparent border-t-blue-200 border-r-blue-200 animate-spin"
                style={{ animationDuration: "3s" }}
              />
              <div
                className="absolute inset-[5px] rounded-full border-2 border-transparent border-t-blue-400 border-l-blue-400 animate-spin"
                style={{ animationDuration: "1.5s", animationDirection: "reverse" }}
              />
              <div
                className="absolute inset-[10px] rounded-full border-2 border-transparent border-t-blue-600 animate-spin"
                style={{ animationDuration: "0.75s" }}
              />
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            </div>

            {/* Equalizer bars */}
            <div className="flex items-end gap-px" style={{ height: "22px" }}>
              {BAR_HEIGHTS.map((h, i) => (
                <div
                  key={i}
                  className="w-1.5 rounded-t-sm animate-bounce"
                  style={{
                    height: `${h}%`,
                    backgroundColor: `rgba(59, 130, 246, ${0.2 + i * 0.08})`,
                    animationDelay: `${i * 65}ms`,
                    animationDuration: `${520 + (i % 4) * 130}ms`,
                  }}
                />
              ))}
            </div>

            {/* Real step from backend */}
            <span className="text-[10px] font-mono text-blue-400 tracking-wide w-full text-center px-2">
              {result.current_step ?? "Initializing..."}
            </span>
          </div>
        )}
      </div>

      <div className="p-3 flex-shrink-0">
        <div className="flex items-center justify-between gap-1">
          <h4 className="text-sm font-semibold text-gray-800 truncate">{result.display_name}</h4>
          {isCompleted && modelName && dataUrl && (
            <a
              href={`/optimizer?task=${task}&model=${encodeURIComponent(modelName)}&explainer=${encodeURIComponent(result.explainer_name)}&data_url=${encodeURIComponent(dataUrl)}`}
              title="Open in Optimizer"
              className="flex-shrink-0 flex items-center gap-0.5 text-[10px] text-green-600 hover:text-green-700 border border-green-200 hover:border-green-300 rounded px-1.5 py-0.5 transition-colors"
            >
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Optimize
            </a>
          )}
        </div>
        {isCompleted && (
          <div className="mt-1.5 grid grid-cols-3 gap-x-1 text-center">
            {metrics.map((m) => {
              const isRanked = activeMetrics.includes(m.key);
              return (
                <div key={m.key} className={`py-0.5 ${isRanked ? "bg-blue-50 rounded" : ""}`}>
                  <p className="text-[10px] text-gray-400 leading-tight">{m.label}</p>
                  <p className="text-xs font-mono font-semibold text-gray-700 leading-tight">
                    {m.value?.toFixed(3) ?? "N/A"}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
