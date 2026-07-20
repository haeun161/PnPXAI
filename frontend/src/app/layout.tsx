import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const freesentation = localFont({
  src: [
    { path: "./fonts/Freesentation-1Thin.ttf", weight: "100", style: "normal" },
    { path: "./fonts/Freesentation-2ExtraLight.ttf", weight: "200", style: "normal" },
    { path: "./fonts/Freesentation-3Light.ttf", weight: "300", style: "normal" },
    { path: "./fonts/Freesentation-4Regular.ttf", weight: "400", style: "normal" },
    { path: "./fonts/Freesentation-5Medium.ttf", weight: "500", style: "normal" },
    { path: "./fonts/Freesentation-6SemiBold.ttf", weight: "600", style: "normal" },
    { path: "./fonts/Freesentation-7Bold.ttf", weight: "700", style: "normal" },
    { path: "./fonts/Freesentation-8ExtraBold.ttf", weight: "800", style: "normal" },
    { path: "./fonts/Freesentation-9Black.ttf", weight: "900", style: "normal" },
  ],
  variable: "--font-freesentation",
  display: "swap",
});

export const metadata: Metadata = {
  title: "XAI Demo Platform",
  description: "Interactive eXplainable AI demo for image classification",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={freesentation.variable}>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
