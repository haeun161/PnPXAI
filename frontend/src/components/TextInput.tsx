"use client";
import { useState, useRef } from "react";

interface Props {
  onTextReady: (blob: Blob, preview: string) => void;
  disabled?: boolean;
}

export default function TextInput({ onTextReady, disabled }: Props) {
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleTextChange = (value: string) => {
    setText(value);
    if (value.trim()) {
      const blob = new Blob([value], { type: "text/plain" });
      onTextReady(blob, value.slice(0, 100) + (value.length > 100 ? "..." : ""));
    }
  };

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result as string;
      setText(content);
      const blob = new Blob([content], { type: "text/plain" });
      onTextReady(blob, content.slice(0, 100) + (content.length > 100 ? "..." : ""));
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-gray-700">Input Text</label>
      <textarea
        value={text}
        onChange={(e) => handleTextChange(e.target.value)}
        placeholder="e.g. This movie was absolutely wonderful and I loved every moment of it!"
        disabled={disabled}
        className="w-full h-32 rounded-lg border border-gray-300 px-3 py-2 text-sm resize-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
      />
      <div className="flex items-center gap-2">
        <button
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="text-xs text-blue-600 hover:text-blue-700 disabled:opacity-50"
        >
          Or upload .txt file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".txt,text/plain"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
