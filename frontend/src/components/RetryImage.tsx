"use client";
import { useEffect, useState } from "react";

interface RetryImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  maxRetries?: number;
  retryDelayMs?: number;
}

// A one-off failed load (e.g. the request racing a busy dev server) otherwise
// leaves the <img> permanently broken since browsers never retry on their own.
export default function RetryImage({ src, maxRetries = 3, retryDelayMs = 800, ...rest }: RetryImageProps) {
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setAttempt(0);
  }, [src]);

  const displaySrc = attempt === 0 ? src : `${src}${src.includes("?") ? "&" : "?"}retry=${attempt}`;

  return (
    <img
      src={displaySrc}
      onError={() => {
        if (attempt < maxRetries) {
          setTimeout(() => setAttempt((a) => a + 1), retryDelayMs * (attempt + 1));
        }
      }}
      {...rest}
    />
  );
}
