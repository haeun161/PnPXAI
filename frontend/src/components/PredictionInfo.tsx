"use client";
import { ForecastInfo, PredictionItem, TaskType } from "@/lib/types";
import { useEffect, useState } from "react";
import RetryImage from "./RetryImage";

interface Props {
  dataUrl: string | null;
  predictions: PredictionItem[] | null;
  forecast?: ForecastInfo | null;
  task: TaskType;
}

const CHART_W = 400;
const CHART_H = 120;
const CHART_PAD = 8;
const N_X_TICKS = 5;
const N_Y_TICKS = 4;

function formatAxisValue(v: number): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// The input window the model saw, followed by the horizon it predicted — with the
// observed values over that same horizon overlaid when the upload was long enough to
// hold them out (a back-test), so predicted and actual can be read against each other.
function ForecastChart({ forecast }: { forecast: ForecastInfo }) {
  const [channel, setChannel] = useState(forecast.attributed_channel);

  const ch = Math.min(channel, forecast.col_names.length - 1);
  const context = forecast.context.map((row) => row[ch]);
  const predicted = forecast.predicted.map((row) => row[ch]);
  const actual = forecast.actual ? forecast.actual.map((row) => row[ch]) : null;

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

  const labels = [...(forecast.context_labels ?? []), ...(forecast.horizon_labels ?? [])];
  const hasLabels = labels.length === n;
  const xTickIndices = Array.from({ length: N_X_TICKS }, (_, i) =>
    Math.round((i * (n - 1)) / (N_X_TICKS - 1))
  );
  const yTickValues = Array.from({ length: N_Y_TICKS }, (_, i) => max - (i * range) / (N_Y_TICKS - 1));

  let mae: number | null = null;
  let mape: number | null = null;
  if (actual) {
    const errors = actual.map((v, i) => v - predicted[i]);
    mae = errors.reduce((s, e) => s + Math.abs(e), 0) / actual.length;
    mape =
      (actual.reduce((s, v, i) => s + (v !== 0 ? Math.abs(errors[i] / v) : 0), 0) / actual.length) * 100;
  }

  return (
    <div className="w-full">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-1.5 text-xs text-gray-600">
        {/* Swatches name the lines; the window lengths get their own group after a
            divider, under the parameter names they are set by. A bare "Input (96)" left
            the reader to guess what the number counted. */}
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
        <span className="text-gray-300">|</span>
        <span>
          seq_len = <span className="font-bold text-gray-900">{ctxLen}</span>
        </span>
        <span>
          pred_len = <span className="font-bold text-gray-900">{predicted.length}</span>
        </span>
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
        <select
          value={ch}
          onChange={(e) => setChannel(Number(e.target.value))}
          className="ml-auto rounded border border-gray-300 bg-white px-1.5 py-0.5 text-xs"
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
          <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} preserveAspectRatio="none" className="w-full h-40">
            {/* Shade the input window so the forecast region reads as distinct. */}
            <rect x={0} y={0} width={xAt(ctxLen - 1)} height={CHART_H} fill="#F3F4F6" />
            <line
              x1={xAt(ctxLen - 1)} y1={0} x2={xAt(ctxLen - 1)} y2={CHART_H}
              stroke="#9CA3AF" strokeWidth="1" strokeDasharray="3,3" vectorEffect="non-scaling-stroke"
            />
            <polyline points={contextPoints} fill="none" stroke="#6B7280" strokeWidth="1.3" vectorEffect="non-scaling-stroke" />
            {actualPoints && (
              <polyline points={actualPoints} fill="none" stroke="#111827" strokeWidth="1.3" vectorEffect="non-scaling-stroke" />
            )}
            <polyline points={predPoints} fill="none" stroke="#FF0000" strokeWidth="2.0" strokeDasharray="4,3" vectorEffect="non-scaling-stroke" />
            {/* Transparent hover targets: a transparent fill is not hit-tested unless
                pointerEvents is forced on. */}
            {context.map((v, i) => (
              <circle key={`c${i}`} cx={xAt(i)} cy={yAt(v)} r="6" fill="transparent" pointerEvents="all">
                <title>{`${hasLabels ? labels[i] : `t=${i}`}\nInput: ${formatAxisValue(v)}`}</title>
              </circle>
            ))}
            {predicted.map((v, i) => (
              <circle key={`p${i}`} cx={xAt(ctxLen + i)} cy={yAt(v)} r="6" fill="transparent" pointerEvents="all">
                <title>
                  {`${hasLabels ? labels[ctxLen + i] : `t=${ctxLen + i}`}\nPredicted: ${formatAxisValue(v)}` +
                    (actual ? `\nActual: ${formatAxisValue(actual[i])}` : "")}
                </title>
              </circle>
            ))}
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

export default function PredictionInfo({ dataUrl, predictions, forecast, task }: Props) {
  const [textContent, setTextContent] = useState<string | null>(null);

  useEffect(() => {
    if (task === "text" && dataUrl) {
      fetch(dataUrl).then((r) => r.text()).then(setTextContent).catch(() => setTextContent(null));
    }
  }, [task, dataUrl]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-4 py-3">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Input & Prediction</h3>
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
              <ForecastChart forecast={forecast} />
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
