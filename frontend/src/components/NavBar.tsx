"use client";
import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV_ITEMS = [
  { label: "Explanation", href: "/" },
  { label: "Optimization", href: "/optimizer" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-0">
      <div className="max-w-[1600px] mx-auto px-6 flex items-center justify-between">
        {/* Brand */}
        {/* items-baseline: a replaced element's baseline is its bottom edge, so the logo's
            bottom lands exactly on the title's baseline — both read as sitting on one line. */}
        <div className="py-4 flex-shrink-0 flex items-baseline gap-3">
          {/* 78x41 native, so h-8 only downscales — it stays crisp */}
          <img
            src="/logo.png"
            alt="PnPXAI logo"
            className="h-8 w-auto select-none"
            draggable={false}
          />
          <h1 className="text-xl font-bold text-gray-900">
            PnPXAI: Plug-and-Play Explainable AI
          </h1>
        </div>

        {/* Nav tabs */}
        <nav className="flex items-end gap-1 self-end">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-5 py-2.5 text-sm font-medium rounded-t-lg border-t border-x transition-colors ${
                  isActive
                    ? "bg-white border-gray-200 text-blue-600 -mb-px border-b-white"
                    : "bg-gray-50 border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
