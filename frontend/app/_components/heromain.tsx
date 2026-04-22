import Link from "next/link";
import {
  ArrowRight,
  Bot,
  Cpu,
  GitBranch,
  Sparkles,
  Target,
  Waves,
} from "lucide-react";

const painPoints = [
  {
    icon: Cpu,
    value: "70%",
    label: "typical GPU waste",
  },
  {
    icon: Target,
    value: "+58%",
    label: "throughput uplift",
  },
  {
    icon: Sparkles,
    value: "<1 min",
    label: "time to first fix",
  },
];

const optimizationTags = [
  { title: "Kernel fusion", delta: "+18%" },
  { title: "Memory layout", delta: "+11%" },
  { title: "Launch config", delta: "+8%" },
  { title: "4 more optimizations", delta: "Queued" },
];

const heatmapRows = [
  [0.22, 0.3, 0.36, 0.42, 0.55, 0.48, 0.34, 0.22],
  [0.24, 0.34, 0.4, 0.62, 0.76, 0.68, 0.46, 0.28],
  [0.18, 0.26, 0.46, 0.74, 0.98, 0.9, 0.58, 0.34],
  [0.12, 0.24, 0.38, 0.66, 0.84, 0.72, 0.5, 0.3],
  [0.08, 0.16, 0.3, 0.44, 0.56, 0.5, 0.38, 0.22],
  [0.05, 0.12, 0.2, 0.3, 0.34, 0.32, 0.24, 0.14],
];

const chartPath =
  "M10 176 C22 166 34 156 46 148 C58 139 70 129 82 120 C94 111 106 103 118 96 C130 89 142 84 154 80 C166 76 178 74 190 69 C202 64 214 57 226 52 C238 47 250 45 262 39 C268 36 272 32 276 28";

const chartAreaPath = `${chartPath} L276 176 L10 176 Z`;

const chartDots = [
  { x: 10, y: 176 },
  { x: 46, y: 148 },
  { x: 82, y: 120 },
  { x: 118, y: 96 },
  { x: 154, y: 80 },
  { x: 190, y: 69 },
  { x: 226, y: 52 },
  { x: 262, y: 39 },
  { x: 276, y: 28, active: true },
];

const chartMetrics = [
  { label: "Utilization", value: "92%" },
  { label: "Queue depth", value: "14 jobs" },
  { label: "Policy drift", value: "Stable" },
];

export default function Hero() {
  return (
    <section className="relative overflow-hidden py-14 sm:py-18 lg:py-20">
      <div className="pointer-events-none absolute inset-x-0 top-14 h-px bg-gradient-to-r from-transparent via-violet-400/25 to-transparent" />
      <div className="pointer-events-none absolute left-[8%] top-24 h-72 w-72 rounded-full bg-violet-500/12 blur-3xl" />
      <div className="pointer-events-none absolute bottom-10 right-[12%] h-80 w-80 rounded-full bg-fuchsia-500/10 blur-3xl" />

      <div className="grid items-start gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(620px,1.05fr)] lg:gap-14">
        <div className="relative z-10 max-w-2xl pt-4 lg:pt-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/35 bg-violet-500/10 px-4 py-2 text-[0.78rem] font-semibold text-violet-200">
            <Sparkles size={14} className="text-violet-300" />
            Open source · Apache 2.0
          </div>

          <h1 className="mt-7 max-w-xl text-5xl font-semibold tracking-[-0.07em] text-white sm:text-6xl lg:text-[5.25rem] lg:leading-[0.95]">
            Stop wasting
            <br />
            <span className="text-violet-400">70% of your GPU.</span>
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300 sm:text-[1.15rem]">
            Profile your training and inference jobs, get the bottleneck named
            for you, and ship the highest-ROI fix - validated by safe
            experiments, not hope.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {painPoints.map((item) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.label}
                  className="flex items-start gap-3 border-l border-white/10 pl-4 text-sm text-slate-300 first:border-l-0 first:pl-0"
                >
                  <span className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-violet-400/25 bg-violet-500/10 text-violet-300">
                    <Icon size={18} />
                  </span>
                  <span>
                    <span className="block leading-6">{item.label}</span>
                    <span className="mt-1 block font-semibold text-white">
                      {item.value}
                    </span>
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Link
              href="#demo"
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-6 py-4 text-sm font-semibold text-white shadow-[0_20px_60px_rgba(124,58,237,0.35)] transition hover:from-violet-500 hover:to-fuchsia-400"
            >
              Start saving compute
              <ArrowRight size={16} />
            </Link>
            <a
              href="https://github.com/fournex/fournex"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/12 bg-white/[0.03] px-6 py-4 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/[0.06]"
            >
              <GitBranch size={16} />
              View on GitHub
            </a>
          </div>

          <div className="mt-10">
            <p className="text-sm text-slate-400">
              Built for production. Works with PyTorch, JAX, TensorFlow, and
              more.
            </p>
            <div className="mt-5 flex flex-wrap items-center gap-x-8 gap-y-4 text-xl font-semibold tracking-[-0.04em] text-slate-500">
              <span>PyTorch</span>
              <span>JAX</span>
              <span>TensorFlow</span>
              <span>NVIDIA</span>
            </div>
          </div>
        </div>

        <div className="relative z-10">
          <div className="absolute -inset-x-4 -inset-y-6 rounded-[2rem] bg-[radial-gradient(circle_at_20%_30%,rgba(124,58,237,0.18),transparent_38%),radial-gradient(circle_at_80%_60%,rgba(217,70,239,0.14),transparent_36%)] blur-2xl" />
          <div className="pointer-events-none absolute inset-0 rounded-[2rem] border border-violet-400/12 bg-[linear-gradient(135deg,rgba(139,92,246,0.06),transparent_36%,transparent_70%,rgba(56,189,248,0.05))]" />
          <div className="relative overflow-hidden rounded-[2rem] border border-violet-400/25 bg-[linear-gradient(180deg,rgba(8,10,24,0.96),rgba(8,10,20,0.9))] p-5 shadow-[0_40px_120px_rgba(5,8,20,0.75)]">
            <div className="flex items-center justify-between border-b border-white/8 pb-4">
              <div>
                <div className="text-base font-semibold text-white">
                  GPU Performance Autopilot
                </div>
                <div className="mt-1 text-xs uppercase tracking-[0.24em] text-slate-500">
                  Continuous optimization plane
                </div>
              </div>
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Live
              </div>
            </div>

            <div className="mt-5 rounded-[1.6rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="text-sm font-medium text-slate-300">
                    Performance improvement
                  </div>
                  <div className="mt-3 text-5xl font-semibold tracking-[-0.06em] text-emerald-300">
                    +41%
                  </div>
                  <div className="mt-2 text-sm text-slate-500">Throughput</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-300">
                    High confidence
                  </div>
                  <div className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs font-medium text-slate-300">
                    RL policy active
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-4 xl:grid-cols-[170px_minmax(0,1fr)]">
                <div className="grid gap-3">
                  {chartMetrics.map((metric) => (
                    <div
                      key={metric.label}
                      className="rounded-[1rem] border border-white/8 bg-[#080b16] px-4 py-3"
                    >
                      <div className="text-[0.65rem] uppercase tracking-[0.22em] text-slate-500">
                        {metric.label}
                      </div>
                      <div className="mt-2 text-sm font-semibold text-white">
                        {metric.value}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="rounded-[1.25rem] border border-white/8 bg-[#070915] px-4 py-5">
                  <div className="pointer-events-none relative h-40 overflow-hidden">
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.06)_1px,transparent_1px)] bg-[size:100%_25%,16.66%_100%]" />
                    <div className="absolute inset-x-0 bottom-0 h-[1px] bg-white/10" />
                    <div className="absolute left-0 top-0 flex h-full flex-col justify-between py-1 text-[0.65rem] text-slate-500">
                      <span>60%</span>
                      <span>40%</span>
                      <span>20%</span>
                      <span>0%</span>
                    </div>
                    <div className="absolute bottom-4 left-8 right-4 top-4">
                      <svg viewBox="0 0 286 152" className="h-full w-full">
                        <defs>
                          <linearGradient id="hero-line" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#8b5cf6" />
                            <stop offset="52%" stopColor="#38bdf8" />
                            <stop offset="100%" stopColor="#22c55e" />
                          </linearGradient>
                          <linearGradient id="hero-area" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="rgba(56,189,248,0.26)" />
                            <stop offset="45%" stopColor="rgba(34,197,94,0.12)" />
                            <stop offset="100%" stopColor="rgba(8,10,24,0)" />
                          </linearGradient>
                          <filter id="hero-glow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="4" result="blur" />
                            <feMerge>
                              <feMergeNode in="blur" />
                              <feMergeNode in="SourceGraphic" />
                            </feMerge>
                          </filter>
                        </defs>
                        <path d={chartAreaPath} fill="url(#hero-area)" />
                        <path
                          d={chartPath}
                          fill="none"
                          stroke="url(#hero-line)"
                          strokeWidth="4"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          filter="url(#hero-glow)"
                        />
                        {chartDots.map((dot) => (
                          <g key={`${dot.x}-${dot.y}`}>
                            <circle
                              cx={dot.x}
                              cy={dot.y}
                              r={dot.active ? 7 : 4}
                              fill={dot.active ? "#22c55e" : "#0b1220"}
                              stroke={dot.active ? "#bbf7d0" : "rgba(255,255,255,0.35)"}
                              strokeWidth={dot.active ? 2.5 : 1.5}
                            />
                            {dot.active ? (
                              <circle
                                cx={dot.x}
                                cy={dot.y}
                                r="12"
                                fill="none"
                                stroke="rgba(34,197,94,0.28)"
                                strokeWidth="1.5"
                              />
                            ) : null}
                          </g>
                        ))}
                      </svg>
                      <div className="absolute left-10 top-3 text-[0.65rem] uppercase tracking-[0.22em] text-slate-500">
                        Throughput trend
                      </div>
                      <div className="absolute right-1 top-3 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1 text-[0.65rem] font-medium text-emerald-300">
                        High confidence
                      </div>
                      <div className="absolute right-0 top-12 rounded-xl border border-white/10 bg-[#0b1120]/90 px-3 py-2 shadow-[0_16px_40px_rgba(3,6,18,0.45)]">
                        <div className="text-[0.6rem] uppercase tracking-[0.22em] text-slate-500">
                          Current uplift
                        </div>
                        <div className="mt-1 text-sm font-semibold text-white">
                          +41% throughput
                        </div>
                      </div>
                    </div>
                    <div className="absolute inset-x-8 bottom-0 flex justify-between text-[0.65rem] text-slate-500">
                      <span>May 12</span>
                      <span>May 19</span>
                      <span>May 26</span>
                      <span>Jun 2</span>
                      <span>Jun 9</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="mb-3 text-sm font-medium text-slate-300">
                Optimizations applied
              </div>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {optimizationTags.map((item, index) => (
                  <div
                    key={item.title}
                    className="rounded-[1.15rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.02))] px-4 py-3"
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={`flex h-9 w-9 items-center justify-center rounded-xl ${
                          index === 0
                            ? "bg-emerald-400/15 text-emerald-300"
                            : index === 1
                              ? "bg-amber-400/15 text-amber-300"
                              : index === 2
                                ? "bg-cyan-400/15 text-cyan-300"
                                : "bg-violet-400/15 text-violet-300"
                        }`}
                      >
                        <Waves size={16} />
                      </span>
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-white">
                          {item.title}
                        </div>
                        <div
                          className={`mt-1 text-xs font-semibold ${
                            index === 3 ? "text-slate-400" : "text-emerald-300"
                          }`}
                        >
                          {item.delta}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
              <div className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-5">
                <div className="text-sm font-medium text-white">
                  Workload profiling
                </div>
                <div className="mt-4 grid gap-4 sm:grid-cols-[0.95fr_1.05fr] sm:items-center">
                  <div className="space-y-3 text-sm">
                    {[
                      ["SM utilization", "62%"],
                      ["Memory BW", "78%"],
                      ["DRAM stalls", "34%"],
                      ["Kernel launches", "1.2M"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex items-center justify-between gap-3 text-slate-400">
                        <span>{label}</span>
                        <span className="font-medium text-white">{value}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div className="rounded-[1rem] border border-white/8 bg-[#080b16] p-3">
                      <div className="grid grid-cols-8 gap-1">
                      {heatmapRows.flatMap((row, rowIndex) =>
                        row.map((cell, cellIndex) => (
                          <span
                            key={`${rowIndex}-${cellIndex}`}
                            className="aspect-square rounded-[4px]"
                            style={{
                              backgroundColor: `rgba(236,72,153,${cell})`,
                              boxShadow:
                                cell > 0.7
                                  ? "0 0 18px rgba(249,115,22,0.25)"
                                  : "none",
                            }}
                          />
                        )),
                      )}
                      </div>
                      <div className="mt-3 flex items-center justify-between text-[0.65rem] uppercase tracking-[0.22em] text-slate-500">
                        <span>Cold</span>
                        <span className="mx-3 h-px flex-1 bg-gradient-to-r from-violet-400 via-fuchsia-500 to-orange-400" />
                        <span>Hot</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.015))] p-5">
                <div className="text-sm font-medium text-white">
                  Autopilot agent (RL)
                </div>
                <div className="relative mt-5 flex h-[220px] items-center justify-center overflow-hidden rounded-[1.25rem] border border-white/8 bg-[#070915]">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(139,92,246,0.1),transparent_45%)]" />
                  <div className="absolute h-28 w-28 rounded-full border border-violet-400/30 bg-violet-500/10" />
                  <div className="absolute h-20 w-20 rounded-full border border-violet-400/25 bg-violet-500/10" />
                  <div className="relative z-10 flex h-16 w-16 items-center justify-center rounded-full border border-violet-400/35 bg-violet-500/15 text-violet-200">
                    <Bot size={28} />
                  </div>
                  <span className="absolute left-8 top-8 text-xs text-slate-400">
                    Observe
                    <br />
                    (Profile)
                  </span>
                  <span className="absolute right-10 top-10 text-right text-xs text-slate-400">
                    Decide
                    <br />
                    (RL / Search)
                  </span>
                  <span className="absolute bottom-8 right-12 text-right text-xs text-slate-400">
                    Act
                    <br />
                    (Apply / Suggest)
                  </span>
                  <span className="absolute bottom-10 left-10 text-xs text-slate-400">
                    Reward
                    <br />
                    (Performance)
                  </span>
                  <svg
                    viewBox="0 0 320 220"
                    className="absolute inset-0 h-full w-full text-white/12"
                    fill="none"
                  >
                    <path d="M160 34C212 34 254 74 254 110" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M254 110C254 148 220 182 160 184" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M160 184C102 184 66 150 66 110" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M66 110C66 72 102 34 160 34" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
