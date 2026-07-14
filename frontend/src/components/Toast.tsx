"use client";
import { useEffect, useState } from "react";

interface Props {
  message: string;
  show: boolean;
  duration?: number;
  onHide: () => void;
}

export default function Toast({ message, show, duration = 2000, onHide }: Props) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!show) return;
    setVisible(true);
    const t = setTimeout(() => {
      setVisible(false);
      setTimeout(onHide, 300);
    }, duration);
    return () => clearTimeout(t);
  }, [show]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!show && !visible) return null;

  return (
    <div className={`fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 pointer-events-none transition-all duration-300 ${visible ? "opacity-100 scale-100" : "opacity-0 scale-95"}`}>
      <div className="flex items-center gap-3 bg-gray-900/90 text-white text-sm font-medium px-5 py-3 rounded-xl shadow-2xl backdrop-blur-sm">
        <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
        </svg>
        {message}
      </div>
    </div>
  );
}
