"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRight, Menu, X } from "lucide-react";

function FournexMark({ size = 36 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
      <rect
        x="0.75"
        y="0.75"
        width="30.5"
        height="30.5"
        rx="9"
        fill="rgba(34,211,238,0.1)"
        stroke="rgba(34,211,238,0.4)"
        strokeWidth="1.5"
      />
      <rect x="7" y="7" width="7" height="7" rx="2" fill="rgb(34,211,238)" />
      <rect
        x="18"
        y="7"
        width="7"
        height="7"
        rx="2"
        fill="rgba(34,211,238,0.45)"
      />
      <rect
        x="7"
        y="18"
        width="7"
        height="7"
        rx="2"
        fill="rgba(34,211,238,0.45)"
      />
      <rect
        x="18"
        y="18"
        width="7"
        height="7"
        rx="2"
        fill="rgba(34,211,238,0.2)"
      />
      <line
        x1="14"
        y1="10.5"
        x2="18"
        y2="10.5"
        stroke="rgba(34,211,238,0.5)"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <line
        x1="10.5"
        y1="14"
        x2="10.5"
        y2="18"
        stroke="rgba(34,211,238,0.5)"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

const navLinks = [
  { href: "/how-it-works", label: "How it works" },
  { href: "/docs", label: "Docs" },
  { href: "/#roadmap", label: "Company" },
];

export default function Header() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full">
      <div className="relative px-4 sm:px-6 lg:px-8 xl:px-10">
        <div className="mx-auto flex min-h-[4.25rem] w-full max-w-[92rem] items-center justify-between rounded-[1.35rem] border border-white/[0.08] bg-black/30 px-4 py-2.5 backdrop-blur-2xl backdrop-saturate-150 shadow-[0_8px_32px_-6px_rgba(0,0,0,0.3)] sm:px-5 lg:px-6 xl:px-7">
          <Link href="/" className="flex items-center gap-3 text-sm font-medium">
            <FournexMark size={32} />
            <span className="text-base font-semibold tracking-[-0.03em] text-white sm:text-lg">
              Fournex
            </span>
          </Link>

          <nav className="hidden items-center gap-6 text-sm text-slate-300 lg:flex lg:gap-8">
            {navLinks.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className="transition-colors duration-200 hover:text-white"
              >
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <Link
              href="/signin"
              className="hidden items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 text-sm font-medium text-white shadow-sm backdrop-blur-sm transition-all duration-200 hover:border-white/20 hover:bg-white/[0.08] hover:shadow-md sm:inline-flex"
            >
              Sign in
            </Link>
            <Link
              href="/analyze"
              className="group relative inline-flex items-center gap-1.5 overflow-hidden rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-4 py-2.5 text-sm font-medium text-white shadow-sm shadow-fuchsia-500/20 transition-all duration-300 hover:shadow-md hover:shadow-fuchsia-500/30"
            >
              <span className="absolute inset-0 bg-gradient-to-r from-violet-500 to-fuchsia-400 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              <span className="relative z-10 flex items-center gap-1.5">
                Try demo
                <ArrowRight size={14} />
              </span>
            </Link>

            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              aria-label="Toggle navigation menu"
              aria-expanded={mobileOpen}
              className="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] p-2.5 text-white transition-all duration-200 hover:border-white/20 hover:bg-white/[0.08] lg:hidden"
            >
              {mobileOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {mobileOpen && (
          <div className="absolute left-4 right-4 top-full z-50 mt-2 overflow-hidden rounded-[1.35rem] border border-white/[0.08] bg-black/90 p-4 shadow-[0_8px_32px_-6px_rgba(0,0,0,0.6)] backdrop-blur-2xl lg:hidden sm:left-6 sm:right-6">
            <nav className="flex flex-col gap-1">
              {navLinks.map((item) => (
                <Link
                  key={item.label}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className="rounded-xl px-4 py-3 text-sm text-slate-300 transition-colors duration-150 hover:bg-white/[0.06] hover:text-white"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="mt-3 flex flex-col gap-2 border-t border-white/8 pt-3">
              <Link
                href="/signin"
                onClick={() => setMobileOpen(false)}
                className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-center text-sm font-medium text-white transition-all duration-200 hover:border-white/20 hover:bg-white/[0.08]"
              >
                Sign in
              </Link>
              <Link
                href="/analyze"
                onClick={() => setMobileOpen(false)}
                className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-4 py-3 text-sm font-medium text-white"
              >
                Try demo
                <ArrowRight size={14} />
              </Link>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
