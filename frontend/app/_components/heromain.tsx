'use client';

import Link from "next/link";
import {
  ArrowRight,
  CircleDot,
  Code2,
  Database,
  Boxes,
  Clock,
  Zap,
} from "lucide-react";

const heroStats = [
  { value: "70%", label: "typical GPU waste" },
  { value: "+58%", label: "throughput uplift" },
  { value: "<1 min", label: "time to first fix" },
];

const detectedBottlenecks = [
  {
    icon: Database,
    label: "Dataloader starvation",
    severity: 82,
    metric: "GPU idle 58%",
    tone: "bg-emerald-400 text-emerald-300 border-emerald-400/30",
  },
  {
    icon: Boxes,
    label: "Small-batch inefficiency",
    severity: 64,
    metric: "Occupancy 31%",
    tone: "bg-cyan-400 text-cyan-300 border-cyan-400/30",
  },
  {
    icon: Clock,
    label: "Host-device sync",
    severity: 48,
    metric: "14 stalls / step",
    tone: "bg-amber-400 text-amber-300 border-amber-400/30",
  },
];

export default function Hero() {
  return (
    <section className="relative w-full py-20 sm:py-24 lg:py-28">
      <div className="grid items-start gap-14 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:gap-16">
        {/* Left: Copy */}
        <div className="flex flex-col items-start gap-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-sm font-medium text-emerald-300">
            <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            GPU Autopilot for PyTorch + NVIDIA
          </div>

          <div className="space-y-6">
            <h1 className="max-w-4xl text-5xl font-semibold tracking-[-0.06em] text-white sm:text-6xl lg:text-[4.75rem] lg:leading-[1.02]">
              Stop wasting{" "}
              <span className="hidden lg:inline">
                <br />
              </span>
              <span className="text-emerald-300">70% of your GPU.</span>
            </h1>

            <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
              Profile your training and inference jobs, get the bottleneck named
              for you, and ship the highest-ROI fix — validated by safe
              experiments, not hope.
            </p>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
            <Link
              href="#demo"
              className="group inline-flex items-center justify-center gap-2 rounded-full bg-emerald-400 px-6 py-3.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300"
            >
              <Zap size={18} />
              Start saving compute
              <ArrowRight
                size={16}
                className="transition group-hover:translate-x-0.5"
              />
            </Link>
            <Link
              href="#how-it-works"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-6 py-3.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/10"
            >
              <Code2 size={18} />
              See a sample report
            </Link>
          </div>

          {/* Stat strip */}
          <dl className="mt-4 grid w-full max-w-xl grid-cols-3 divide-x divide-white/10 rounded-2xl border border-white/10 bg-white/[0.03]">
            {heroStats.map((stat) => (
              <div key={stat.label} className="px-5 py-4">
                <dt className="text-[0.65rem] font-medium uppercase tracking-[0.22em] text-slate-500">
                  {stat.label}
                </dt>
                <dd className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-white">
                  {stat.value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        {/* Right: Bottleneck report card */}
        <div className="relative w-full">
          <div className="relative rounded-[1.75rem] border border-white/10 bg-slate-950/80 shadow-2xl">
            {/* Card header */}
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 text-cyan-300">
                  <CircleDot size={14} />
                </div>
                <div>
                  <div className="text-xs font-medium text-white">
                    resnet50_train · node-14
                  </div>
                  <div className="font-mono text-[0.65rem] text-slate-500">
                    trace · 14c2b · 18.2s
                  </div>
                </div>
              </div>
              <span className="rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-[0.6rem] font-medium uppercase tracking-[0.2em] text-emerald-300">
                Analyzed
              </span>
            </div>

            {/* Top summary row */}
            <div className="grid grid-cols-3 divide-x divide-white/10 border-b border-white/10">
              <div className="px-5 py-4">
                <div className="text-[0.6rem] uppercase tracking-[0.22em] text-slate-500">
                  GPU active
                </div>
                <div className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-white">
                  42%
                </div>
              </div>
              <div className="px-5 py-4">
                <div className="text-[0.6rem] uppercase tracking-[0.22em] text-slate-500">
                  Uplift
                </div>
                <div className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-emerald-300">
                  +68%
                </div>
              </div>
              <div className="px-5 py-4">
                <div className="text-[0.6rem] uppercase tracking-[0.22em] text-slate-500">
                  Monthly save
                </div>
                <div className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-white">
                  $12.4k
                </div>
              </div>
            </div>

            {/* Bottleneck rows with solid severity bars */}
            <ul className="divide-y divide-white/10">
              {detectedBottlenecks.map((b) => {
                const Icon = b.icon;
                const [fillColor, textColor, borderColor] = b.tone.split(" ");
                return (
                  <li key={b.label} className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div
                        className={`flex h-9 w-9 items-center justify-center rounded-xl border ${borderColor} bg-white/[0.03] ${textColor}`}
                      >
                        <Icon size={16} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-3">
                          <div className="truncate text-sm font-medium text-white">
                            {b.label}
                          </div>
                          <div className="font-mono text-xs text-slate-400">
                            {b.metric}
                          </div>
                        </div>
                        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
                          <div
                            className={`h-full rounded-full ${fillColor}`}
                            style={{ width: `${b.severity}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>

            {/* Footer action */}
            <div className="flex items-center justify-between border-t border-white/10 px-5 py-4">
              <div className="text-xs text-slate-400">
                <span className="font-mono text-emerald-300">3 fixes</span>{" "}
                ranked, ready to validate
              </div>
              <button
                type="button"
                className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-medium text-white transition hover:border-white/30 hover:bg-white/10"
              >
                Apply top 3
                <ArrowRight size={12} />
              </button>
            </div>
          </div>

          {/* Floating badge */}
          <div className="absolute -bottom-5 -right-3 hidden rounded-2xl border border-white/10 bg-slate-950 p-4 shadow-2xl md:block">
            <div className="flex items-center gap-3">
              <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/10 p-2 text-emerald-300">
                <Zap size={16} />
              </div>
              <div>
                <div className="text-[0.65rem] uppercase tracking-[0.22em] text-slate-500">
                  Validated
                </div>
                <div className="text-sm font-semibold text-white">
                  +58% faster
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
