import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Boxes,
  CheckCircle2,
  CircuitBoard,
  Clock,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  LineChart,
  ListChecks,
  Microscope,
  Play,
  Radar,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Workflow,
  Zap,
} from "lucide-react";
import Hero from "./_components/heromain";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fournex | GPU performance profiler and autopilot",
  description: "Fournex helps you optimize GPU performance with advanced profiling and autopilot features.",
};

// ── Fournex logo mark ──────────────────────────────────────────────────────
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

// ── Data ───────────────────────────────────────────────────────────────────

const trustLogos = [
  "AI21 labs",
  "Perplexity",
  "character.ai",
  "Midjourney",
  "Cohere",
  "Anyscale",
];

const bottleneckPatterns = [
  {
    icon: Database,
    title: "Dataloader starvation",
    symptom: "GPU idle while CPU workers stall",
    detail:
      "Detect when the input pipeline can't feed the device. Tune workers, prefetching, and pinned memory before re-running.",
    tone: "emerald",
  },
  {
    icon: Boxes,
    title: "Small-batch inefficiency",
    symptom: "Low SM occupancy under real batch sizes",
    detail:
      "Pinpoint shapes that under-utilize kernels. Recommend batching, padding, or recompile-safe shape hints.",
    tone: "cyan",
  },
  {
    icon: Clock,
    title: "Host-device sync overhead",
    symptom: "Frequent .item() and blocking transfers",
    detail:
      "Find hidden synchronization stalls — scalar reads, premature .cpu() calls, and chatty copy patterns.",
    tone: "amber",
  },
  {
    icon: Sparkles,
    title: "Mixed-precision opportunities",
    symptom: "FP32 where AMP is safe and faster",
    detail:
      "Identify layers and ops that can move to bf16/fp16 without loss of accuracy, ranked by expected speedup.",
    tone: "violet",
  },
  {
    icon: CircuitBoard,
    title: "Kernel launch fragmentation",
    symptom: "Many tiny kernels, launch overhead dominates",
    detail:
      "Spot fusion candidates for torch.compile, CUDA graphs, or Triton — with concrete before/after projections.",
    tone: "sky",
  },
  {
    icon: Radar,
    title: "Memory pressure & fragmentation",
    symptom: "OOM risk, swaps, cache thrashing",
    detail:
      "Track allocator behavior, fragmentation, and layout choices that silently cap throughput ceiling.",
    tone: "rose",
  },
];

const toneMap: Record<
  string,
  { ring: string; bg: string; text: string; dot: string }
> = {
  emerald: {
    ring: "ring-blue-400/20",
    bg: "bg-blue-400/10",
    text: "text-blue-300",
    dot: "bg-blue-400",
  },
  cyan: {
    ring: "ring-cyan-400/20",
    bg: "bg-cyan-400/10",
    text: "text-cyan-300",
    dot: "bg-cyan-400",
  },
  amber: {
    ring: "ring-amber-400/20",
    bg: "bg-amber-400/10",
    text: "text-amber-300",
    dot: "bg-amber-400",
  },
  violet: {
    ring: "ring-violet-400/20",
    bg: "bg-violet-400/10",
    text: "text-violet-300",
    dot: "bg-violet-400",
  },
  sky: {
    ring: "ring-sky-400/20",
    bg: "bg-sky-400/10",
    text: "text-sky-300",
    dot: "bg-sky-400",
  },
  rose: {
    ring: "ring-rose-400/20",
    bg: "bg-rose-400/10",
    text: "text-rose-300",
    dot: "bg-rose-400",
  },
};

const rankedFixes = [
  {
    rank: "01",
    title: "Increase dataloader workers 4 → 12, enable pinned memory",
    effort: "Low",
    confidence: "High",
    speedup: "+28%",
    risk: "Low",
  },
  {
    rank: "02",
    title: "Increase batch size from inferred baseline if memory allows",
    effort: "Low",
    confidence: "High",
    speedup: "+14%",
    risk: "Low",
  },
  {
    rank: "03",
    title: 'Try torch.compile(mode="reduce-overhead") for launch overhead',
    effort: "Medium",
    confidence: "Medium",
    speedup: "+14%",
    risk: "Medium",
  },
  {
    rank: "04",
    title: "Reduce .item() / host sync points in the training loop",
    effort: "Low",
    confidence: "High",
    speedup: "+6%",
    risk: "Low",
  },
];

const productPhases = [
  {
    phase: "Phase 1",
    title: "Profiler with opinions",
    status: "Shipping now",
    icon: Microscope,
    bullets: [
      "Low-overhead trace collection",
      "Deterministic bottleneck classifier",
      "Normalized performance IR",
    ],
  },
  {
    phase: "Phase 2",
    title: "Ranked recommendations",
    status: "Shipping now",
    icon: ListChecks,
    bullets: [
      "Map bottlenecks to concrete fixes",
      "Score by impact, effort, and risk",
      "Explainable, repeatable output",
    ],
  },
  {
    phase: "Phase 3",
    title: "Experiment runner",
    status: "Early access",
    icon: FlaskConical,
    bullets: [
      "Safe config sweeps",
      "Before/after benchmark validation",
      "Regression guardrails",
    ],
  },
  {
    phase: "Phase 4",
    title: "Policy-driven autopilot",
    status: "On the roadmap",
    icon: Workflow,
    bullets: [
      "Auto-apply within your guardrails",
      "Continuous adaptation",
      "Learned optimization policies",
    ],
  },
];

const capabilityCards = [
  {
    icon: Gauge,
    title: "Production profiling",
    description:
      "Low-overhead telemetry for kernels, streams, allocators, and memory traffic.",
  },
  {
    icon: ListChecks,
    title: "Ranked fixes",
    description:
      "Concrete, explainable recommendations — scored by impact, effort, and risk.",
  },
  {
    icon: FlaskConical,
    title: "Safe experiment runner",
    description:
      "Sweep configs safely, validate every trial, and stop bad runs early.",
  },
  {
    icon: ShieldCheck,
    title: "Regression guardrails",
    description:
      "Throughput, memory, loss divergence, and NaN checks — so trust comes built in.",
  },
  {
    icon: TrendingUp,
    title: "Before/after validation",
    description:
      "Every applied change ships with an auditable benchmark delta and reproducibility hash.",
  },
  {
    icon: GitBranch,
    title: "CI-native",
    description:
      "Run as a CLI, in CI, or as a continuous agent. No hardware changes, no vendor lock-in.",
  },
];

const metrics = [
  { value: "Up to 35%", label: "lower GPU cost" },
  { value: "20–50%", label: "throughput gains" },
  { value: "Weeks → hours", label: "faster tuning cycles" },
  { value: "0 new hardware", label: "required for ROI" },
];

const footerLinks = [
  {
    heading: "Product",
    items: ["How it works", "Capabilities", "Pricing", "Changelog"],
  },
  {
    heading: "Resources",
    items: ["Docs", "Benchmarks", "Blog", "Security"],
  },
  {
    heading: "Community",
    items: ["GitHub", "Discord", "Contributing", "Roadmap"],
  },
];

// ── Shared components ──────────────────────────────────────────────────────

function SectionHeading({
  eyebrow,
  title,
  description,
  align = "center",
}: {
  eyebrow: string;
  title: string;
  description: string;
  align?: "center" | "left";
}) {
  const alignment = align === "center" ? "mx-auto text-center" : "text-left";
  const eyebrowAlignment = align === "center" ? "self-center" : "self-start";
  return (
    <div className={`${alignment} flex w-full max-w-3xl flex-col gap-4`}>
      <span
        className={`inline-flex items-center gap-2 ${eyebrowAlignment} text-xs font-semibold uppercase tracking-[0.32em] text-cyan-300/80`}
      >
        <span className="h-px w-8 bg-cyan-300/40" />
        {eyebrow}
      </span>
      <h2 className="text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl lg:text-[2.6rem] lg:leading-[1.1]">
        {title}
      </h2>
      <p className="text-base leading-7 text-slate-300 sm:text-lg">
        {description}
      </p>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(109,40,217,0.26),_transparent_24%),radial-gradient(circle_at_85%_18%,_rgba(59,130,246,0.14),_transparent_20%),linear-gradient(180deg,_#03040b_0%,_#070815_38%,_#04050d_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.05)_1px,transparent_1px)] bg-[size:120px_120px] [mask-image:radial-gradient(circle_at_top,black,transparent_80%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top,_rgba(124,58,237,0.24),_transparent_58%)] blur-3xl" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-[92rem] flex-col px-4 sm:px-8 lg:px-10 xl:px-12">

        {/* ── Hero ── */}
          <Hero />

        {/* ── Trust band ── */}
        <section className="py-6 sm:py-8">
          <div className="rounded-[1.5rem] border border-white/8 bg-white/[0.02] px-4 py-6 sm:rounded-[1.75rem] sm:px-8 sm:py-7">
            <p className="text-center text-sm font-medium leading-6 text-violet-200/85 sm:leading-normal">
              Trusted by teams shipping the world&apos;s most compute-intensive
              workloads
            </p>
            <div className="mt-6 grid grid-cols-2 gap-x-4 gap-y-4 text-center text-base font-semibold tracking-[-0.03em] text-slate-500 sm:grid-cols-3 sm:gap-x-6 sm:gap-y-5 sm:text-xl lg:grid-cols-6">
              {trustLogos.map((item) => (
                <div key={item}>{item}</div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Bottleneck showcase ── */}
        <section id="bottlenecks" className="py-16 sm:py-24 lg:py-28">
          <SectionHeading
            eyebrow="What we detect"
            title="Six GPU bottleneck families. Named, ranked, and fixable."
            description="We skip the raw profiler dumps and go straight to diagnosis. Every bottleneck maps to a concrete, explainable fix; not another dashboard to stare at."
          />
          <div className="mt-10 grid gap-4 sm:mt-14 sm:gap-5 md:grid-cols-2 xl:grid-cols-3">
            {bottleneckPatterns.map((item) => {
              const tone = toneMap[item.tone];
              const Icon = item.icon;
              return (
                <article
                  key={item.title}
                  className="group relative overflow-hidden rounded-[1.25rem] border border-white/8 bg-white/[0.04] p-5 transition hover:border-white/20 hover:bg-white/[0.06] sm:rounded-[1.5rem] sm:p-6"
                >
                  <div
                    className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl ring-1 ${tone.ring} ${tone.bg} ${tone.text}`}
                  >
                    <Icon size={20} />
                  </div>
                  <div className="mt-5 flex items-center gap-2">
                    <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
                    <span className={`text-xs font-medium uppercase tracking-[0.22em] ${tone.text}`}>
                      {item.symptom}
                    </span>
                  </div>
                  <h3 className="mt-3 text-xl font-semibold tracking-[-0.02em] text-white">
                    {item.title}
                  </h3>
                  <p className="mt-3 text-sm leading-7 text-slate-300">
                    {item.detail}
                  </p>
                  <div className="mt-6 inline-flex items-center gap-1.5 text-xs text-slate-400 transition group-hover:text-white">
                    See sample trace
                    <ArrowUpRight size={14} />
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── Ranked recommendations ── */}
        <section id="how-it-works" className="py-16 sm:py-24 lg:py-28">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:items-center">
            <div>
              <SectionHeading
                align="left"
                eyebrow="Ranked recommendations"
                title="From trace to ranked fix list in under a minute."
                description="Every finding comes with expected impact, implementation effort, confidence level, and blast radius. Tune mode can then benchmark selected candidates before you trust the change."
              />
              <ul className="mt-8 grid gap-3 text-sm text-slate-300">
                {[
                  "Explainable, rule-based classifier — no hallucinations",
                  "Ranked by diagnosis strength, expected impact, effort, and risk",
                  "Recommendations separated from benchmarked tune candidates",
                  "Tune mode validates selected configs with guardrails",
                ].map((item) => (
                  <li key={item} className="flex items-start gap-3">
                    <span className="mt-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-400/15 text-blue-300">
                      <CheckCircle2 size={14} />
                    </span>
                    {item}
                  </li>
                ))}
              </ul>
              <div className="mt-10 flex flex-col items-stretch gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                <a
                  href="#demo"
                  className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-blue-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 sm:w-auto"
                >
                  <Play size={16} />
                  See a live report
                </a>
                <a
                  href="#roadmap"
                  className="inline-flex w-full items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm text-white transition hover:border-white/20 hover:bg-white/10 sm:w-auto"
                >
                  Product roadmap
                  <ArrowRight size={14} />
                </a>
              </div>
            </div>

            {/* Mockup card */}
            <div className="relative">
              <div className="pointer-events-none absolute -inset-6 rounded-[2.25rem] bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.14),transparent_60%)] blur-2xl" />
              <div className="relative overflow-hidden rounded-[1.75rem] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.95),rgba(2,6,23,0.9))] shadow-[0_30px_100px_rgba(2,6,23,0.55)]">
                <div className="flex flex-col gap-3 border-b border-white/8 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
                  <div className="flex min-w-0 items-center gap-2 text-xs text-slate-400">
                    <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-400/10 text-cyan-300">
                      <BarChart3 size={14} />
                    </div>
                    <span className="min-w-0 break-all font-mono">trace-14c2b / resnet50_train</span>
                  </div>
                  <span className="rounded-full border border-blue-400/20 bg-blue-400/10 px-3 py-1 text-[0.65rem] font-medium uppercase tracking-[0.2em] text-blue-300">
                    Analyzed
                  </span>
                </div>
                <div className="grid grid-cols-1 divide-y divide-white/8 border-b border-white/8 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
                  {[
                    { label: "GPU active", value: "42%" },
                    { label: "Potential uplift", value: "+68%" },
                    { label: "Est. monthly save", value: "$12.4k" },
                  ].map((stat) => (
                    <div key={stat.label} className="p-4">
                      <div className="text-[0.65rem] uppercase tracking-[0.22em] text-slate-500">
                        {stat.label}
                      </div>
                      <div className="mt-1 text-lg font-semibold tracking-[-0.03em] text-white sm:text-xl">
                        {stat.value}
                      </div>
                    </div>
                  ))}
                </div>
                <ul className="divide-y divide-white/8">
                  {rankedFixes.map((fix) => (
                    <li
                      key={fix.rank}
                      className="flex flex-col items-start gap-4 p-4 transition hover:bg-white/[0.03] sm:flex-row sm:items-center sm:p-5"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/5 font-mono text-xs text-slate-400">
                        {fix.rank}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium text-white">
                          {fix.title}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.7rem] text-slate-400">
                          <span>Effort · {fix.effort}</span>
                          <span>Confidence · {fix.confidence}</span>
                          <span>Risk · {fix.risk}</span>
                        </div>
                      </div>
                      <div className="text-left sm:text-right">
                        <div className="text-sm font-semibold text-blue-300">
                          {fix.speedup}
                        </div>
                        <div className="text-[0.65rem] uppercase tracking-[0.2em] text-slate-500">
                          speedup
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
                <div className="flex flex-col gap-2 border-t border-white/8 bg-white/[0.02] px-4 py-3 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between sm:px-5">
                  <span className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
                    Live recommendation stream
                  </span>
                  <span className="break-all font-mono">report_7a1.json</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Roadmap ── */}
        <section id="roadmap" className="py-16 sm:py-24 lg:py-28">
          <SectionHeading
            eyebrow="Product evolution"
            title="Profiler today. Autopilot tomorrow. Trust built in at every step."
            description="We don't ship blind automation. The product walks from diagnosis to optimization to full autopilot — each phase validated by real workload outcomes before the next one turns on."
          />
          <div className="mt-10 grid gap-4 sm:mt-14 sm:gap-5 md:grid-cols-2 xl:grid-cols-4">
            {productPhases.map((phase, index) => {
              const Icon = phase.icon;
              const isLive = phase.status === "Shipping now";
              const isAccess = phase.status === "Early access";
              return (
                <article
                  key={phase.phase}
                  className="relative flex flex-col overflow-hidden rounded-[1.25rem] border border-white/8 bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(2,6,23,0.7))] p-5 sm:rounded-[1.5rem] sm:p-6"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[0.65rem] font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
                      {phase.phase}
                    </span>
                    <span
                      className={`rounded-full px-3 py-1 text-[0.6rem] font-medium uppercase tracking-[0.18em] ${
                        isLive
                          ? "border border-blue-400/20 bg-blue-400/10 text-blue-300"
                          : isAccess
                            ? "border border-cyan-400/20 bg-cyan-400/10 text-cyan-300"
                            : "border border-white/10 bg-white/[0.04] text-slate-400"
                      }`}
                    >
                      {phase.status}
                    </span>
                  </div>
                  <div className="mt-6 flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white">
                    <Icon size={20} />
                  </div>
                  <h3 className="mt-5 text-xl font-semibold tracking-[-0.02em] text-white">
                    {phase.title}
                  </h3>
                  <ul className="mt-4 flex flex-1 flex-col gap-2 text-sm text-slate-300">
                    {phase.bullets.map((b) => (
                      <li key={b} className="flex items-start gap-2">
                        <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-cyan-300/70" />
                        {b}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-6 h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                  <div className="mt-3 text-[0.65rem] font-mono uppercase tracking-[0.28em] text-slate-500">
                    0{index + 1} · closed loop
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── Capabilities ── */}
        <section id="capabilities" className="py-16 sm:py-24 lg:py-28">
          <SectionHeading
            eyebrow="Capabilities"
            title="Built for platform engineers operating production GPU fleets."
            description="Telemetry fidelity, explainable ranking, and controlled rollout — the product surface is designed for technical teams, not executive dashboards."
          />
          <div className="mt-10 grid gap-4 sm:mt-14 sm:gap-5 md:grid-cols-2 xl:grid-cols-3">
            {capabilityCards.map((item) => {
              const Icon = item.icon;
              return (
                <article
                  key={item.title}
                  className="group rounded-[1.25rem] border border-white/8 bg-white/[0.04] p-5 transition hover:border-cyan-400/20 hover:bg-white/[0.06] sm:rounded-[1.5rem] sm:p-6"
                >
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-white/10 to-white/[0.02] text-cyan-200">
                    <Icon size={20} />
                  </div>
                  <h3 className="mt-5 text-lg font-semibold text-white">
                    {item.title}
                  </h3>
                  <p className="mt-3 text-sm leading-7 text-slate-300">
                    {item.description}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── CLI integration ── */}
        <section className="py-16 sm:py-24 lg:py-28">
          <div className="grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
            <div>
              <SectionHeading
                align="left"
                eyebrow="Drop in. Go."
                title="Collect, diagnose, and race safe fixes from the CLI."
                description="Works as a CLI, a CI job, or a long-running agent. Bring your own cluster, keep your own training code. The tuner screens cheap candidates first, then fully validates only the strongest ones."
              />
              <div className="mt-8 grid gap-3 text-sm sm:gap-4">
                {[
                  { label: "Install", value: "pip install -e ./backend/python" },
                  { label: "Collect", value: "frx collect -- python train.py" },
                  { label: "Tune", value: "frx tune --no-safe --race-promote-count 3 -- python train.py" },
                ].map((step) => (
                  <div
                    key={step.label}
                    className="flex flex-col items-start gap-2 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 sm:flex-row sm:items-center sm:gap-4"
                  >
                    <span className="text-[0.65rem] font-semibold uppercase tracking-[0.24em] text-cyan-300/80">
                      {step.label}
                    </span>
                    <code className="break-all font-mono text-sm text-slate-200">
                      {step.value}
                    </code>
                  </div>
                ))}
              </div>
            </div>

            {/* Terminal */}
            <div className="relative">
              <div className="pointer-events-none absolute -inset-6 rounded-[2.25rem] bg-[radial-gradient(circle_at_bottom_left,rgba(99,102,241,0.18),transparent_60%)] blur-2xl" />
              <div className="relative overflow-hidden rounded-[1.5rem] border border-white/10 bg-slate-950/80 font-mono text-xs shadow-[0_30px_100px_rgba(2,6,23,0.55)] backdrop-blur-xl sm:rounded-[1.75rem] sm:text-sm">
                <div className="flex items-center border-b border-white/10 bg-white/5 px-4 py-3">
                  <div className="flex gap-2">
                    <div className="h-3 w-3 rounded-full bg-red-500/80" />
                    <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
                    <div className="h-3 w-3 rounded-full bg-green-500/80" />
                  </div>
                  <div className="mx-auto text-xs text-slate-400">
                    frx ~ tune.sh
                  </div>
                </div>
                <div className="space-y-3 p-4 leading-6 sm:p-6 sm:leading-7">
                  <div>
                    <span className="text-slate-500">$</span>{" "}
                    <span className="text-cyan-300">frx</span>{" "}
                    <span className="text-slate-200">tune</span>{" "}
                    <span className="text-amber-300">--no-safe</span>{" "}
                    <span className="text-amber-300">--race-promote-count</span>{" "}
                    <span className="text-slate-400">3</span>{" "}
                    <span className="text-amber-300">--</span>{" "}
                    <span className="text-slate-400">python train.py</span>
                  </div>
                  <div className="text-slate-400">
                    <span className="text-blue-300">✓</span> baseline captured:
                    4.80 steps/sec, batch_size=32
                  </div>
                  <div className="text-slate-400">
                    <span className="text-blue-300">✓</span> generated 8 focused
                    candidates from input_bound diagnosis
                  </div>
                  <div className="rounded-xl border border-amber-400/20 bg-amber-400/5 p-3 text-amber-200">
                    <div className="text-xs uppercase tracking-[0.2em] text-amber-300/80">
                      Quick race stage
                    </div>
                    <div className="mt-1 text-slate-100">
                      8 short trials screened down to 3 full benchmarks
                    </div>
                  </div>
                  <div className="space-y-1 text-slate-400">
                    <div>
                      <span className="text-blue-300">✓</span> [RACE]
                      dl:nw=8,pin=T{" "}
                      <span className="text-blue-300">+18.1%</span> promoted
                    </div>
                    <div>
                      <span className="text-blue-300">✓</span> [RACE] bs:40{" "}
                      <span className="text-blue-300">+14.0%</span> promoted
                    </div>
                    <div>
                      <span className="text-blue-300">✓</span> [FULL]
                      dl:nw=8,pin=T,pf=4{" "}
                      <span className="text-blue-300">+24.7%</span>
                    </div>
                  </div>
                  <div className="rounded-xl border border-blue-400/20 bg-blue-400/10 p-3 text-blue-100">
                    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                      <span>Winner from full benchmark</span>
                      <span className="font-semibold">+24.7%</span>
                    </div>
                    <div className="mt-1 text-xs text-blue-200/70">
                      race results cannot promote directly · no loss divergence · no OOM
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── ROI ── */}
        <section id="roi" className="py-16 sm:py-24 lg:py-28">
          <div className="rounded-[1.5rem] border border-white/8 bg-[linear-gradient(135deg,rgba(8,15,31,0.94),rgba(15,23,42,0.8))] p-6 sm:rounded-[2rem] sm:p-10 lg:p-14">
            <SectionHeading
              eyebrow="ROI"
              title="Immediate infrastructure leverage. Not a long science project."
              description="Cut wasted GPU spend, increase throughput, and shrink the manual tuning backlog. No migrations. No new hardware. Just measurable deltas in production."
            />
            <div className="mt-10 grid gap-4 sm:mt-14 sm:gap-5 md:grid-cols-2 xl:grid-cols-4">
              {metrics.map((item) => (
                <div
                  key={item.label}
                  className="group relative overflow-hidden rounded-[1.5rem] border border-white/8 bg-black/30 p-6"
                >
                  <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-cyan-400/10 blur-3xl transition group-hover:bg-cyan-400/20" />
                  <div className="relative text-3xl font-semibold tracking-[-0.05em] text-white sm:text-4xl">
                    {item.value}
                  </div>
                  <p className="relative mt-3 text-sm uppercase tracking-[0.2em] text-slate-400">
                    {item.label}
                  </p>
                </div>
              ))}
            </div>
            <div className="mt-10 flex flex-col items-start gap-4 rounded-[1.25rem] border border-blue-400/15 bg-blue-400/[0.06] p-5 sm:rounded-[1.5rem] sm:p-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="max-w-xl text-sm leading-7 text-blue-50/90">
                A team running a $1.2M/yr GPU training budget typically recovers
                <span className="font-semibold text-white">
                  {" "}$280k–$420k
                </span>{" "}
                in the first quarter after onboarding — before any code changes
                beyond recommended ones.
              </div>
              <a
                href="#demo"
                className="inline-flex items-center gap-2 self-start rounded-full border border-blue-400/30 bg-blue-400/10 px-4 py-2 text-sm text-blue-100 transition hover:bg-blue-400/20 sm:self-auto"
              >
                Model your savings
                <ArrowRight size={14} />
              </a>
            </div>
          </div>
        </section>

        {/* ── Moat ── */}
        <section className="py-16 sm:py-24 lg:py-28">
          <div className="grid gap-8 sm:gap-10 lg:grid-cols-[0.95fr_1.05fr]">
            <div>
              <SectionHeading
                align="left"
                eyebrow="Moat"
                title="A workload-performance dataset that compounds every week."
                description="We're not another rules engine. Every analyzed workload expands the mapping from trace patterns to validated fixes — turning usage into defensibility."
              />
              <div className="mt-8 grid gap-3 text-sm leading-6 text-slate-300">
                {[
                  "Proprietary trace → fix → outcome dataset",
                  "Validated optimization deltas across hardware",
                  "Policies that improve as more teams onboard",
                  "Trust layer: every change auditable and reversible",
                ].map((item) => (
                  <div
                    key={item}
                    className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3"
                  >
                    <LineChart size={16} className="mt-0.5 text-cyan-300" />
                    {item}
                  </div>
                ))}
              </div>
            </div>
            <div className="relative overflow-hidden rounded-[1.5rem] border border-white/8 bg-white/[0.04] p-5 sm:rounded-[2rem] sm:p-7">
              <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-indigo-500/10 blur-3xl" />
              <div className="relative grid gap-4">
                {[
                  "More workloads analyzed",
                  "Richer optimization traces",
                  "Better policy learning",
                  "Stronger recommendations",
                  "More workloads onboard",
                ].map((item, index, arr) => (
                  <div key={item} className="flex items-center gap-4">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 text-xs font-semibold text-cyan-200">
                      0{index + 1}
                    </div>
                    <div className="relative flex-1 rounded-2xl border border-white/8 bg-slate-950/70 px-4 py-3 text-sm text-slate-200">
                      {item}
                      {index < arr.length - 1 && (
                        <span className="absolute -bottom-3 left-6 h-3 w-px bg-cyan-400/30" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="relative mt-6 rounded-[1.5rem] border border-blue-400/15 bg-blue-400/10 p-5">
                <p className="text-sm leading-7 text-blue-50/90">
                  Better data → better policies → better outcomes → more
                  workloads. That loop is the product.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section id="demo" className="py-16 sm:py-24 lg:py-28">
          <div className="relative overflow-hidden rounded-[1.5rem] border border-cyan-400/20 bg-[linear-gradient(135deg,rgba(6,182,212,0.18),rgba(14,165,233,0.08),rgba(15,23,42,0.92))] p-6 sm:rounded-[2rem] sm:p-10 lg:p-14">
            <div className="pointer-events-none absolute -right-16 -top-16 h-72 w-72 rounded-full bg-blue-400/10 blur-3xl" />
            <div className="pointer-events-none absolute -left-24 bottom-0 h-72 w-72 rounded-full bg-indigo-400/10 blur-3xl" />
            <div className="relative grid gap-10 lg:grid-cols-[1fr_0.75fr] lg:items-end">
              <div className="max-w-3xl">
                <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-cyan-200">
                  <Zap size={12} />
                  Early access · Open source
                </span>
                <h2 className="mt-5 text-3xl font-semibold tracking-[-0.05em] text-white sm:text-5xl lg:text-[3.2rem] lg:leading-[1.05]">
                  Turn wasted GPU compute into measurable performance gains.
                </h2>
                <p className="mt-5 max-w-2xl text-base leading-7 text-slate-200 sm:text-lg sm:leading-8">
                  See where your GPU efficiency leaks today, what can be tuned
                  automatically, and how quickly those gains can land in
                  production. First report is free.
                </p>
                <div className="mt-8 grid gap-3 text-sm text-slate-300 sm:flex sm:flex-wrap sm:items-center sm:gap-6">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-blue-300" />
                    No code changes to onboard
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-blue-300" />
                    Works with PyTorch + NVIDIA today
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-blue-300" />
                    Apache 2.0 license
                  </div>
                </div>
              </div>
              <form className="rounded-[1.5rem] border border-white/10 bg-slate-950/80 p-5 backdrop-blur sm:rounded-[1.75rem] sm:p-6">
                <label className="block text-sm text-slate-300">
                  Work email
                  <input
                    type="email"
                    placeholder="team@company.com"
                    className="mt-3 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40 focus:bg-white/[0.07]"
                  />
                </label>
                <label className="mt-4 block text-sm text-slate-300">
                  Primary workload
                  <input
                    type="text"
                    placeholder="Training, inference, simulation…"
                    className="mt-3 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40 focus:bg-white/[0.07]"
                  />
                </label>
                <label className="mt-4 block text-sm text-slate-300">
                  Monthly GPU spend
                  <select
                    className="mt-3 w-full appearance-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-white outline-none transition focus:border-cyan-400/40 focus:bg-white/[0.07]"
                    defaultValue=""
                  >
                    <option value="" disabled className="bg-slate-900">
                      Select a range
                    </option>
                    <option className="bg-slate-900">Under $25k</option>
                    <option className="bg-slate-900">$25k – $100k</option>
                    <option className="bg-slate-900">$100k – $500k</option>
                    <option className="bg-slate-900">$500k+</option>
                  </select>
                </label>
                <button
                  type="submit"
                  className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-full bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
                >
                  Request early access
                  <ArrowRight size={14} />
                </button>
                <p className="mt-4 text-xs leading-6 text-slate-500">
                  We&apos;ll reply within one business day with next steps.
                </p>
              </form>
            </div>
          </div>
        </section>

        {/* ── Footer ── */}
        <footer className="mt-10 border-t border-white/8 py-12">
          <div className="grid gap-8 sm:gap-10 lg:grid-cols-[1.4fr_repeat(3,1fr)]">
            <div>
              <Link href="/" className="flex items-center gap-3">
                <FournexMark size={28} />
                <span className="text-sm font-semibold tracking-[-0.01em] text-white">
                  Fournex
                </span>
              </Link>
              <p className="mt-4 max-w-xs text-sm leading-7 text-slate-400">
                Open-source GPU performance optimizer for PyTorch + NVIDIA.
                From profiler to closed-loop autopilot.
              </p>
              <div className="mt-5 flex items-center gap-3">
                <a
                  href="https://github.com/fournex/fournex"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 transition hover:border-white/20 hover:text-white"
                >
                  <GitBranch size={14} />
                </a>
              </div>
            </div>
            {footerLinks.map((group) => (
              <div key={group.heading}>
                <h4 className="text-[0.7rem] font-semibold uppercase tracking-[0.28em] text-slate-500">
                  {group.heading}
                </h4>
                <ul className="mt-4 space-y-3 text-sm text-slate-300">
                  {group.items.map((item) => (
                    <li key={item}>
                      <a href="#" className="transition hover:text-white">
                        {item}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div className="mt-10 flex flex-col items-start justify-between gap-3 border-t border-white/8 pt-6 text-xs leading-6 text-slate-500 sm:flex-row sm:items-center">
            <span>© {new Date().getFullYear()} Fournex. Open source under Apache 2.0.</span>
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <a href="#" className="transition hover:text-white">Privacy</a>
              <a href="#" className="transition hover:text-white">Terms</a>
              <a href="#" className="transition hover:text-white">Status</a>
            </div>
          </div>
        </footer>

      </div>
    </main>
  );
}
