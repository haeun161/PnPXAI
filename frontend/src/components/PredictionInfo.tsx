"use client";
import { ForecastInfo, PredictionItem, TaskType } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";
import RetryImage from "./RetryImage";

interface Props {
  dataUrl: string | null;
  predictions: PredictionItem[] | null;
  forecast?: ForecastInfo | null;
  task: TaskType;
  // Time-series only: clicking a chained forecast window re-runs the explanation
  // anchored on that window's own input instead of the default (earliest) one.
  // null clears the selection and brings the whole chain back (the Reset button).
  onWindowExplain?: (windowIndex: number | null) => void;
  windowExplainDisabled?: boolean;
}

const CHART_W = 400;
const CHART_H = 120;
const CHART_PAD = 8;
const N_X_TICKS = 5;
const N_Y_TICKS = 4;
// Above this many plotted points, per-timestep hover circles are skipped -- each one
// is a DOM node with its own tooltip. The backend already caps the back-test region to
// a modest size, so this is just a safety net, not something normally hit.
const MAX_HOVER_POINTS = 400;
// Visible dots stop earlier than hover targets do: past this density they merge into a
// solid band and stop reading as individual samples, which is the only reason to draw
// them. The lines alone carry the shape from there on.
const MAX_DOT_POINTS = 160;

// Marker sizes in *rendered* pixels, converted to user units at draw time (see `scale`).
const DOT_R = 2.2;
const TRI_W = 10;
const TRI_H = 8;
const TRI_GAP = 4;

function formatAxisValue(v: number): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// The input window the model saw, followed by the horizon it predicted — with the
// observed values over that same horizon overlaid when the upload was long enough to
// hold them out (a back-test), so predicted and actual can be read against each other.
// The backend caps the chained horizon to a small recent slice, so this always fits in
// one view -- no scrolling or zooming needed.
function ForecastChart({
  forecast,
  onWindowExplain,
  disabled,
}: {
  forecast: ForecastInfo;
  // null clears the selection and brings the whole chain back (the Reset button).
  onWindowExplain?: (windowIndex: number | null) => void;
  disabled?: boolean;
}) {
  const [channel, setChannel] = useState(forecast.attributed_channel);
  const [hoveredWindow, setHoveredWindow] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const ch = Math.min(channel, forecast.col_names.length - 1);

  // preserveAspectRatio="none" stretches the viewBox by a different factor on each axis,
  // so a circle drawn in user units renders as an ellipse and the marker triangle comes
  // out skewed -- and by how much depends on the card's current width. Measuring the
  // rendered box lets each marker be sized in real pixels and divided back into user
  // units, so it keeps its shape at any width. Lines don't need this; they already use
  // vectorEffect="non-scaling-stroke".
  const [scale, setScale] = useState({ x: 1, y: 1 });
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const update = () => {
      const box = el.getBoundingClientRect();
      if (box.width > 0 && box.height > 0) {
        setScale({ x: box.width / CHART_W, y: box.height / CHART_H });
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Hover state (updated on every mousemove) must not re-derive all of this every
  // time, so it's memoized on the data itself (forecast + channel) and nothing else.
  const chart = useMemo(() => {
    const context = forecast.context.map((row) => row[ch]);
    const fullPredicted = forecast.predicted.map((row) => row[ch]);
    const fullActual = forecast.actual ? forecast.actual.map((row) => row[ch]) : null;
    const fullHorizonLabels = forecast.horizon_labels ?? [];

    // `predicted` is a chain of `windowLen`-sized forecasts, each re-fed the real
    // context right before it, covering the back-test region the backend computed.
    const windowLen =
      forecast.window_len && forecast.window_len > 0 ? forecast.window_len : fullPredicted.length;
    const totalWindows = windowLen > 0 ? Math.round(fullPredicted.length / windowLen) : 1;
    // Which window `context` (and the explainers) actually describes -- everything
    // before it is dropped below so the chart never shows two windows' worth of
    // history/predictions mixed together; only this window's own history onward.
    const absoluteExplainedIdx = Math.min(forecast.explained_window_index ?? 0, totalWindows - 1);
    const trimStart = absoluteExplainedIdx * windowLen;
    // Two display modes. By default the whole chain is drawn, so there is something to
    // pick from. Once a window is picked, only that window's own horizon is drawn --
    // one input window and the pred_len it produced, which is exactly what the
    // attributions below describe. Reset goes back to the chain.
    const selected = forecast.window_selected ?? false;
    const trimEnd = selected ? trimStart + windowLen : fullPredicted.length;
    const predicted = fullPredicted.slice(trimStart, trimEnd);
    const actual = fullActual ? fullActual.slice(trimStart, trimEnd) : null;
    const horizonLabels = fullHorizonLabels.slice(trimStart, trimEnd);
    // How many windows remain visible after trimming -- the explained one is always
    // display-relative window 0 now.
    const maxWindows = selected ? 1 : totalWindows - absoluteExplainedIdx;

    const ctxLen = context.length;
    const n = ctxLen + predicted.length;

    const allValues = [...context, ...predicted, ...(actual ?? [])];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    const stepX = (CHART_W - CHART_PAD * 2) / Math.max(n - 1, 1);
    const xAt = (i: number) => CHART_PAD + i * stepX;
    const yAt = (v: number) => CHART_H - CHART_PAD - ((v - min) / range) * (CHART_H - CHART_PAD * 2);
    const pt = (i: number, v: number) => `${xAt(i)},${yAt(v)}`;

    // Both horizon lines start at the last context point so they read as continuations
    // of the input rather than as free-floating segments.
    const anchor = pt(ctxLen - 1, context[ctxLen - 1]);
    const contextPoints = context.map((v, i) => pt(i, v)).join(" ");
    const predPoints = [anchor, ...predicted.map((v, i) => pt(ctxLen + i, v))].join(" ");
    const actualPoints = actual
      ? [anchor, ...actual.map((v, i) => pt(ctxLen + i, v))].join(" ")
      : null;

    // Mark where every chained window ends and the next (re-fed with real context)
    // begins.
    const windowBoundaries = Array.from({ length: maxWindows - 1 }, (_, i) => ctxLen + (i + 1) * windowLen);

    const labels = [...(forecast.context_labels ?? []), ...horizonLabels];
    const hasLabels = labels.length === n;
    const xTickIndices = Array.from({ length: N_X_TICKS }, (_, i) =>
      Math.round((i * (n - 1)) / (N_X_TICKS - 1))
    );
    const yTickValues = Array.from({ length: N_Y_TICKS }, (_, i) => max - (i * range) / (N_Y_TICKS - 1));

    // Scored over the *whole* back-test chain, not just what's currently displayed --
    // otherwise trimming to a later window (or clicking around) would change the
    // number shown, when it's meant to read as "how good is this model overall,"
    // the same regardless of which window happens to be selected.
    let mae: number | null = null;
    let mape: number | null = null;
    if (fullActual) {
      const errors = fullActual.map((v, i) => v - fullPredicted[i]);
      mae = errors.reduce((s, e) => s + Math.abs(e), 0) / fullActual.length;
      mape =
        (fullActual.reduce((s, v, i) => s + (v !== 0 ? Math.abs(errors[i] / v) : 0), 0) / fullActual.length) * 100;
    }

    return {
      context, predicted, actual, windowLen, maxWindows, absoluteExplainedIdx, selected, ctxLen, n,
      stepX, xAt, yAt, contextPoints, predPoints, actualPoints, windowBoundaries, labels, hasLabels,
      xTickIndices, yTickValues, mae, mape,
    };
  }, [forecast, ch]);
  const {
    context, predicted, actual, windowLen, maxWindows, absoluteExplainedIdx, selected, ctxLen, n,
    stepX, xAt, yAt, contextPoints, predPoints, actualPoints, windowBoundaries, labels, hasLabels,
    xTickIndices, yTickValues, mae, mape,
  } = chart;

  // Converts a pointer's clientX into a window index by way of the SVG's rendered
  // bounding box -- one overlay covering every window, rather than one per window.
  // Nothing left to pick once a window is selected -- the chart is showing that one
  // window alone, and clicking it would just re-run the identical job. Reset is the way
  // back, and it stays enabled.
  const resettable = !!onWindowExplain && !disabled;
  const clickable = resettable && !selected;
  const windowIndexAt = (clientX: number): number | null => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || box.width === 0) return null;
    const vbX = ((clientX - box.left) / box.width) * CHART_W;
    const w = Math.floor((((vbX - CHART_PAD) / stepX) - ctxLen) / windowLen);
    return w >= 0 && w < maxWindows ? w : null;
  };
  const windowRect = (w: number) => {
    // Window 0's predicted/actual lines are anchored back at the last context point
    // (ctxLen - 1), one step before its own painted region -- extend its highlight
    // that same one step so it meets the gray history shading with no visible gap.
    const x1 = xAt(w === 0 ? ctxLen - 1 : ctxLen + w * windowLen);
    const x2 = xAt(ctxLen + (w + 1) * windowLen);
    return { x: x1, width: Math.max(x2 - x1, 0) };
  };

  const showPointHovers = n <= MAX_HOVER_POINTS;
  const showDots = n <= MAX_DOT_POINTS;

  // Round dots and an unskewed triangle, in user units (see `scale` above).
  const dotRx = DOT_R / scale.x;
  const dotRy = DOT_R / scale.y;

  // The explainers attribute one scalar: the first step of the explained window's
  // horizon, on the attributed channel. That is exactly display index `ctxLen`, so the
  // marker can point at the real datum rather than at the window as a whole. It is
  // hidden on any other channel -- nothing was attributed there, and an unlabelled
  // arrow would read as if something had been.
  const marker = useMemo(() => {
    if (ch !== forecast.attributed_channel || predicted.length === 0) return null;
    const cx = xAt(ctxLen);
    const py = yAt(predicted[0]);
    const halfW = TRI_W / 2 / scale.x;
    const h = TRI_H / scale.y;
    const gap = TRI_GAP / scale.y;
    // Sitting above the point would clip it off the top whenever that point is the
    // series maximum -- which, for a forecast's first step, is not unusual. Flip below
    // and point up instead; either way the tip touches the datum.
    const above = py - gap - h >= 0;
    const tipY = above ? py - gap : py + gap;
    const baseY = above ? tipY - h : tipY + h;
    return {
      points: `${cx - halfW},${baseY} ${cx + halfW},${baseY} ${cx},${tipY}`,
      value: predicted[0],
      label: hasLabels ? labels[ctxLen] : `t=${ctxLen}`,
    };
  }, [ch, forecast.attributed_channel, predicted, xAt, yAt, ctxLen, scale, hasLabels, labels]);

  return (
    <div className="w-full">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-1.5 text-sm text-gray-600">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-1 bg-gray-400" /> Input
        </span>
        {actual && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-1 bg-gray-900" /> Actual
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-0 border-t-2 border-dashed border-red-500" aria-hidden />
          Predicted
        </span>
        {marker && (
          <span
            className="flex items-center gap-1.5"
            title="The single value the explainers attribute: the first predicted step of the explained window"
          >
            <span className="text-blue-600 leading-none text-[10px]" aria-hidden>▼</span>
            Attribution target
          </span>
        )}
        <span className="text-gray-300">|</span>
        <span>
          seq_len = <span className="font-bold text-gray-900">{ctxLen}</span>
        </span>
        <span>
          pred_len = <span className="font-bold text-gray-900">{windowLen}</span>
        </span>
        {maxWindows > 1 && (
          <span>
            windows = <span className="font-bold text-gray-900">{maxWindows}</span>
          </span>
        )}
        {mae !== null && mape !== null && (
          <>
            <span className="text-gray-300">|</span>
            <span>
              MAE <span className="font-bold text-gray-900">{formatAxisValue(mae)}</span>
            </span>
            <span>
              MAPE <span className="font-bold text-gray-900">{mape.toFixed(1)}%</span>
            </span>
          </>
        )}
        {selected && (
          <button
            onClick={() => { if (resettable) onWindowExplain!(null); }}
            disabled={!resettable}
            title="Show every forecast window again, so a different one can be picked"
            className="flex items-center gap-1 rounded border border-gray-300 bg-white px-1.5 py-0.5 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h5M4 9a8 8 0 1 1 2 5.3" />
            </svg>
            Reset
          </button>
        )}
        <select
          value={ch}
          onChange={(e) => setChannel(Number(e.target.value))}
          className="ml-auto rounded border border-gray-300 bg-white px-1.5 py-0.5 text-sm"
        >
          {forecast.col_names.map((name, i) => (
            <option key={name} value={i}>
              {name}
              {i === forecast.attributed_channel ? " (explained)" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="flex gap-1.5">
        <div className="flex flex-col justify-between text-xs text-gray-500 h-40 py-0.5 flex-shrink-0">
          {yTickValues.map((v, i) => (
            <span key={i}>{formatAxisValue(v)}</span>
          ))}
        </div>
        <div className="flex-1 min-w-0">
          <svg ref={svgRef} viewBox={`0 0 ${CHART_W} ${CHART_H}`} preserveAspectRatio="none" className="w-full h-40">
            {/* Shade the explained window's own history/lookback so it reads as
                distinct from what it predicted -- moves with whichever window is
                currently explained. */}
            <rect x={0} y={0} width={xAt(ctxLen - 1)} height={CHART_H} fill="#F3F4F6" />
            <line
              x1={xAt(ctxLen - 1)} y1={0} x2={xAt(ctxLen - 1)} y2={CHART_H}
              stroke="#9CA3AF" strokeWidth="1" strokeDasharray="3,3" vectorEffect="non-scaling-stroke"
            />
            {/* Boundaries between chained pred_len windows within the forecast. */}
            {windowBoundaries.map((idx) => (
              <line
                key={`wb${idx}`}
                x1={xAt(idx)} y1={0} x2={xAt(idx)} y2={CHART_H}
                stroke="#E5E7EB" strokeWidth="1" strokeDasharray="2,2" vectorEffect="non-scaling-stroke"
              />
            ))}
            {/* Persistent + hover highlights -- just the 1-2 windows that need it,
                never one rect per window. The explained one is always display
                window 0 -- everything before it was trimmed off above. */}
            <rect {...windowRect(0)} y={0} height={CHART_H} fill="#3B82F6" fillOpacity={0.12} pointerEvents="none" />
            {hoveredWindow !== null && hoveredWindow !== 0 && (
              <rect {...windowRect(hoveredWindow)} y={0} height={CHART_H} fill="#3B82F6" fillOpacity={0.06} pointerEvents="none" />
            )}
            <polyline points={contextPoints} fill="none" stroke="#6B7280" strokeWidth="1.3" vectorEffect="non-scaling-stroke" />
            {actualPoints && (
              <polyline points={actualPoints} fill="none" stroke="#111827" strokeWidth="1.3" vectorEffect="non-scaling-stroke" />
            )}
            <polyline points={predPoints} fill="none" stroke="#FF0000" strokeWidth="2.0" strokeDasharray="4,3" vectorEffect="non-scaling-stroke" />
            {/* Sample dots, in each line's own colour. The dashed predicted line in
                particular hides where one step ends and the next begins; the dots put
                the individual timesteps back. pointerEvents off so they never take a
                click meant for the window overlay below. */}
            {showDots && (
              <g pointerEvents="none">
                {context.map((v, i) => (
                  <ellipse key={`cd${i}`} cx={xAt(i)} cy={yAt(v)} rx={dotRx} ry={dotRy} fill="#6B7280" />
                ))}
                {actual?.map((v, i) => (
                  <ellipse key={`ad${i}`} cx={xAt(ctxLen + i)} cy={yAt(v)} rx={dotRx} ry={dotRy} fill="#111827" />
                ))}
                {predicted.map((v, i) => (
                  <ellipse key={`pd${i}`} cx={xAt(ctxLen + i)} cy={yAt(v)} rx={dotRx} ry={dotRy} fill="#FF0000" />
                ))}
              </g>
            )}
            {/* Everything hit-testable lives under this group, and the window
                click/hover handlers sit on the group rather than on the overlay rect.
                The per-point tooltip targets below are wide enough (r=6 against a
                ~5-unit step) to blanket the predicted region, and they paint after the
                overlay -- so SVG hit-testing hands them the click. As siblings of the
                rect they would swallow it; as children of this group they bubble into
                the same handler and clicking a window keeps working anywhere in the
                region, tooltip or not. */}
            <g
              onMouseMove={(e) => setHoveredWindow(clickable ? windowIndexAt(e.clientX) : null)}
              onMouseLeave={() => setHoveredWindow(null)}
              onClick={(e) => {
                const w = windowIndexAt(e.clientX);
                // windowIndexAt is relative to what's currently displayed (already
                // trimmed to start at absoluteExplainedIdx) -- convert back to an
                // absolute window index before asking to re-explain it.
                if (w !== null && clickable) onWindowExplain!(absoluteExplainedIdx + w);
              }}
            >
              {/* Base hit area for the gaps between points. */}
              <rect
                x={xAt(ctxLen)} y={0} width={Math.max(xAt(n - 1) - xAt(ctxLen), 0)} height={CHART_H}
                fill="transparent"
                pointerEvents={clickable ? "all" : "none"}
                style={{ cursor: clickable ? "pointer" : "default" }}
              />
              {/* Transparent hover targets: a transparent fill is not hit-tested unless
                  pointerEvents is forced on. */}
              {showPointHovers && context.map((v, i) => (
                <circle key={`c${i}`} cx={xAt(i)} cy={yAt(v)} r="6" fill="transparent" pointerEvents="all">
                  <title>{`${hasLabels ? labels[i] : `t=${i}`}\nInput: ${formatAxisValue(v)}`}</title>
                </circle>
              ))}
              {showPointHovers && predicted.map((v, i) => (
                <circle
                  key={`p${i}`} cx={xAt(ctxLen + i)} cy={yAt(v)} r="6" fill="transparent"
                  pointerEvents="all" style={{ cursor: clickable ? "pointer" : "default" }}
                >
                  <title>
                    {`${hasLabels ? labels[ctxLen + i] : `t=${ctxLen + i}`}\nPredicted: ${formatAxisValue(v)}` +
                      (actual ? `\nActual: ${formatAxisValue(actual[i])}` : "")}
                  </title>
                </circle>
              ))}
              {/* Drawn last so it stays legible over the lines and the window highlight. */}
              {marker && (
                <polygon
                  points={marker.points} fill="#2563EB" stroke="#FFFFFF" strokeWidth="0.5"
                  pointerEvents="all" style={{ cursor: clickable ? "pointer" : "default" }}
                >
                  <title>
                    {`Attribution target\n${marker.label}\n` +
                      `Predicted ${forecast.col_names[ch]}: ${formatAxisValue(marker.value)}\n\n` +
                      `Every attribution below is the gradient of this one value ` +
                      `with respect to the input window.`}
                  </title>
                </polygon>
              )}
            </g>
          </svg>
          <div className="flex justify-between text-xs text-gray-500 mt-0.5">
            {xTickIndices.map((idx, i) => (
              <span key={i}>{hasLabels ? labels[idx] : idx}</span>
            ))}
          </div>
        </div>
      </div>
      {!actual && (
        <p className="mt-1 text-xs text-gray-400">
          Forecast runs past the end of the uploaded data — no observed values to compare against.
        </p>
      )}
    </div>
  );
}

export default function PredictionInfo({ dataUrl, predictions, forecast, task, onWindowExplain, windowExplainDisabled }: Props) {
  const [textContent, setTextContent] = useState<string | null>(null);

  useEffect(() => {
    if (task === "text" && dataUrl) {
      fetch(dataUrl).then((r) => r.text()).then(setTextContent).catch(() => setTextContent(null));
    }
  }, [task, dataUrl]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-4 py-3">
      <h3 className="text-base font-semibold text-gray-700 mb-2">Input & Prediction</h3>
      <div className={`flex gap-4 ${task === "timeseries" && forecast ? "flex-col" : ""}`}>
        {/* Original data display */}
        {task === "image" && dataUrl && (
          <RetryImage src={dataUrl} alt="Original" className="max-h-40 w-40 flex-shrink-0 rounded-lg object-contain border" />
        )}
        {task === "text" && (
          <div className="max-h-40 w-56 overflow-y-auto rounded-lg border bg-gray-50 p-2.5 text-sm text-gray-800 flex-shrink-0">
            {textContent || "Loading..."}
          </div>
        )}
        {task === "timeseries" &&
          (forecast ? (
            <div className="w-full rounded-lg border bg-gray-50 p-2">
              <ForecastChart forecast={forecast} onWindowExplain={onWindowExplain} disabled={windowExplainDisabled} />
            </div>
          ) : (
            <div className="w-56 h-24 rounded-lg border bg-gray-50 flex items-center justify-center text-xs text-gray-400 flex-shrink-0">
              Time-series data loaded
            </div>
          ))}

        {/* Predictions */}
        {predictions && predictions.length > 0 && (
          <div className="flex-1 space-y-1.5">
            <p className="text-xs font-medium text-gray-500 uppercase">Top-3 Predictions</p>
            {predictions.slice(0, 3).map((pred, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex-1">
                  <div className="flex justify-between text-sm">
                    <span className={i === 0 ? "font-semibold text-blue-700" : "text-gray-700"}>
                      {pred.class_name}
                    </span>
                    <span className="text-gray-500">{pred.probability.toFixed(1)}%</span>
                  </div>
                  <div className="mt-0.5 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${i === 0 ? "bg-blue-500" : "bg-gray-300"}`}
                      style={{ width: `${Math.min(pred.probability, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
