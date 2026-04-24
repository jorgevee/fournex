import type { Metadata } from "next";
import { Activity, BarChart3, Gauge, Network, Sparkles } from "lucide-react";
import { AnalyzerClient } from "./_components/analyzer-client";

export const metadata: Metadata = {
  title: "Analyzer | Fournex",
  description:
    "Paste GPU telemetry or workflow run summaries and get a clear bottleneck diagnosis with concrete highest-ROI fixes.",
};

const stats = [
  { label: "Accepted inputs", value: "Trace JSON", icon: Network },
  { label: "Output", value: "Diagnosis + fixes", icon: BarChart3 },
  { label: "Feedback", value: "Built in", icon: Gauge },
];

export default async function AnalyzePage() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#05070d] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.05)_1px,transparent_1px)] bg-[size:96px_96px] [mask-image:radial-gradient(circle_at_top,black,transparent_78%)]" />
      <div className="relative mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
        <section className="grid gap-8 lg:grid-cols-[0.86fr_1.14fr] lg:items-start">
          <div className="lg:sticky lg:top-24">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-3 py-1 text-sm font-medium text-cyan-200">
              <Activity size={14} />
              Autopilot Analyzer
            </div>

            <h1 className="mt-5 max-w-2xl text-4xl font-semibold tracking-tight text-white sm:text-5xl">
              Find the bottleneck and ship the next measured fix.
            </h1>
            <p className="mt-4 max-w-xl text-base leading-7 text-slate-300">
              Paste a Fournex summary, raw event trace, or workflow timing JSON. The analyzer separates diagnosis from action so the next step is clear.
            </p>

            <div className="mt-8 grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
              {stats.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.label}
                    className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20"
                  >
                    <div className="flex items-center gap-3">
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-cyan-400/20 bg-cyan-400/10 text-cyan-200">
                        <Icon size={17} />
                      </span>
                      <div>
                        <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          {item.label}
                        </p>
                        <p className="mt-0.5 text-sm font-medium text-slate-100">{item.value}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-8 rounded-lg border border-white/10 bg-white/[0.035] p-5">
              <div className="flex items-start gap-3">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-emerald-400/10 text-emerald-300">
                  <Sparkles size={16} />
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">Structured output</p>
                  <p className="mt-1 text-sm leading-6 text-slate-400">
                    The report keeps primary diagnosis, supporting evidence, and fixes in separate sections so users can scan without rereading the same label.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <AnalyzerClient />
        </section>
      </div>
    </main>
  );
}
