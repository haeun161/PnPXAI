"use client";
import { useState, useRef } from "react";

interface Props {
  onDataReady: (blob: Blob, preview: string) => void;
  disabled?: boolean;
}

export default function TimeSeriesInput({ onDataReady, disabled }: Props) {
  const [values, setValues] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleValuesChange = (v: string) => {
    setValues(v);
    setFileName(null);
    if (v.trim()) {
      const csv = "value\n" + v.split(",").map((x) => x.trim()).filter(Boolean).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      onDataReady(blob, `${v.split(",").length} values`);
    }
  };

  const handleFile = (file: File) => {
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result as string;
      setValues("");
      const blob = new Blob([content], { type: "text/csv" });
      const lines = content.trim().split("\n").length;
      onDataReady(blob, `${file.name} (${lines} rows)`);
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-2">
      {/* no label - tab already indicates this is upload */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="text-xs px-3 py-1.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-50"
        >
          Upload CSV
        </button>
        {fileName && <span className="text-xs text-gray-500">{fileName}</span>}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
