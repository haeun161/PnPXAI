"use client";
import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV_ITEMS = [
  { label: "Analysis", href: "/" },
  { label: "Optimizer", href: "/optimizer" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-0">
      <div className="max-w-[1600px] mx-auto px-6 flex items-center justify-between">
        {/* Brand */}
        <div className="py-4 flex-shrink-0">
          <h1 className="text-xl font-bold text-gray-900">
            XAI Demo Platform <span className="text-sm font-normal text-blue-600">v2</span>
          </h1>
          <p className="text-sm text-gray-500">Multi-modal eXplainable AI powered by PnPXAI</p>
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
