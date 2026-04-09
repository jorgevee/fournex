import Link from "next/link";

const trustCategories = [
  "AI infra",
  "Robotics",
  "Simulation",
  "Gaming",
  "Quant systems",
  "HPC",
];

const problemPoints = [
  {
    title: "GPU spend keeps rising",
    description:
      "Teams scale training and inference faster than they can tune workloads, turning cluster growth into a budget problem.",
  },
  {
    title: "Manual tuning does not scale",
    description:
      "Kernel fusion, launch parameters, and memory layouts still depend on a small pool of CUDA specialists.",
  },
  {
    title: "Observability stops at diagnosis",
    description:
      "Most tools expose hotspots and stalls, but engineering teams still carry the burden of fixing them by hand.",
  },
];

const workflowSteps = [
  {
    step: "01",
    title: "Profile live GPU workloads",
    description:
      "Capture kernel timing, occupancy, memory traffic, and launch behavior directly from production environments.",
  },
  {
    step: "02",
    title: "Detect bottlenecks automatically",
    description:
      "Identify memory divergence, underutilized kernels, launch inefficiencies, and throughput cliffs without manual investigation.",
  },
  {
    step: "03",
    title: "Recommend or apply fixes",
    description:
      "Generate optimization actions across kernel fusion, memory layouts, launch configuration, and scheduling policies.",
  },
  {
    step: "04",
    title: "Improve with search and RL",
    description:
      "Continuously learn from workload outcomes so optimization policies compound in quality over time.",
  },
];

const capabilityCards = [
  {
    title: "Production profiling",
    description:
      "Low-overhead telemetry for kernels, streams, allocators, and memory traffic across your fleet.",
  },
  {
    title: "Kernel optimization recommendations",
    description:
      "Surface concrete changes around fusion, launch shape, thread/block sizing, and execution paths.",
  },
  {
    title: "Memory access analysis",
    description:
      "Pinpoint coalescing issues, cache misses, transfers, and layout choices that degrade effective throughput.",
  },
  {
    title: "Launch config tuning",
    description:
      "Search parameter space for occupancy, latency, and throughput gains instead of relying on static defaults.",
  },
  {
    title: "Continuous learning engine",
    description:
      "Use search and reinforcement learning to evolve optimization strategies as workload patterns change.",
  },
  {
    title: "Safe rollout controls",
    description:
      "Gate optimizations with review flows, staged deployment, fallback policies, and human-in-the-loop approval.",
  },
  {
    title: "Fleet-wide optimization insights",
    description:
      "Track recurring bottlenecks, highest-value opportunities, and realized savings across teams and clusters.",
  },
];

const metrics = [
  { value: "Up to 35%", label: "lower GPU cost" },
  { value: "20-50%", label: "throughput gains" },
  { value: "Weeks to hours", label: "faster tuning cycles" },
  { value: "0 new hardware", label: "required for ROI" },
];

const useCases = [
  "AI training",
  "Inference serving",
  "Simulation",
  "Robotics",
  "Gaming engines",
  "Quant and HPC",
];

const moatSteps = [
  "More workloads analyzed",
  "Richer optimization traces",
  "Better policy learning",
  "Stronger recommendations",
];

const testimonials = [
  {
    quote:
      "We had profiling coverage already. What we lacked was a system that could turn those traces into performance actions automatically.",
    author: "Platform Engineering Lead",
    team: "Large-scale inference team",
  },
  {
    quote:
      "The value is not another dashboard. It is closing the loop between bottleneck detection and measurable GPU savings.",
    author: "VP of Infrastructure",
    team: "Simulation and robotics company",
  },
];

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 text-center lg:text-left">
      <span className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
        {eyebrow}
      </span>
      <h2 className="text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">
        {title}
      </h2>
      <p className="text-base leading-7 text-slate-300 sm:text-lg">
        {description}
      </p>
    </div>
  );
}

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.18),_transparent_26%),radial-gradient(circle_at_80%_20%,_rgba(129,140,248,0.16),_transparent_22%),linear-gradient(180deg,_#020617_0%,_#050816_42%,_#02030a_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:96px_96px] [mask-image:radial-gradient(circle_at_center,black,transparent_82%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-64 bg-[radial-gradient(circle_at_top,_rgba(45,212,191,0.22),_transparent_60%)] blur-3xl" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-6 sm:px-8 lg:px-10">
        <header className="sticky top-0 z-20 -mx-6 border-b border-white/8 bg-slate-950/70 px-6 backdrop-blur-xl sm:-mx-8 sm:px-8 lg:-mx-10 lg:px-10">
          <div className="mx-auto flex h-18 w-full max-w-7xl items-center justify-between">
            <Link href="/" className="flex items-center gap-3 text-sm font-medium">
              <span className="flex h-10 w-10 items-center justify-center rounded-2xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-200 shadow-[0_0_30px_rgba(56,189,248,0.15)]">
                GP
              </span>
              <span className="text-sm tracking-[0.12em] text-slate-100 uppercase">
                GPU Performance Autopilot
              </span>
            </Link>
            <nav className="hidden items-center gap-8 text-sm text-slate-300 md:flex">
              <a href="#how-it-works" className="transition hover:text-white">
                How it works
              </a>
              <a href="#capabilities" className="transition hover:text-white">
                Capabilities
              </a>
              <a href="#use-cases" className="transition hover:text-white">
                Use cases
              </a>
              <a href="#roi" className="transition hover:text-white">
                ROI
              </a>
            </nav>
            <div className="flex items-center gap-3">
              <a
                href="#demo"
                className="hidden rounded-full border border-white/12 px-4 py-2 text-sm text-slate-200 transition hover:border-white/25 hover:bg-white/6 sm:inline-flex"
              >
                Book a demo
              </a>
              <a
                href="#demo"
                className="inline-flex rounded-full bg-white px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-100"
              >
                Request access
              </a>
            </div>
          </div>
        </header>

        <section className="relative flex flex-col justify-center py-20 sm:py-24 lg:min-h-[calc(100vh-4.5rem)] lg:py-12">
          <div className="grid items-center gap-14 xl:gap-18 lg:grid-cols-[1fr_1.08fr]">
            <div className="max-w-3xl">
              <span className="inline-flex rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-cyan-200">
                Autonomous GPU optimization
              </span>
              <h1 className="mt-8 max-w-4xl text-5xl font-semibold tracking-[-0.06em] text-white sm:text-6xl lg:text-7xl">
                Put Your GPU Fleet on Autopilot.
              </h1>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
                Automatically profile, tune, and improve CUDA workloads in
                production. GPU Performance Autopilot turns wasted compute into
                lower spend, higher throughput, and faster engineering loops.
              </p>

              <div className="mt-10 flex flex-col gap-4 sm:flex-row">
                <a
                  href="#demo"
                  className="inline-flex items-center justify-center rounded-full bg-cyan-400 px-6 py-3.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300"
                >
                  Book a demo
                </a>
                <a
                  href="#how-it-works"
                  className="inline-flex items-center justify-center rounded-full border border-white/12 bg-white/5 px-6 py-3.5 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/8"
                >
                  See how it works
                </a>
              </div>

              <div className="mt-10 grid gap-4 sm:grid-cols-3">
                <div className="rounded-3xl border border-white/8 bg-white/5 p-5 backdrop-blur-sm">
                  <div className="text-2xl font-semibold text-white">30-70%</div>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    of GPU compute often goes unused or under-optimized.
                  </p>
                </div>
                <div className="rounded-3xl border border-white/8 bg-white/5 p-5 backdrop-blur-sm">
                  <div className="text-2xl font-semibold text-white">RL + search</div>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    compounds improvements instead of freezing tuning logic in
                    static rules.
                  </p>
                </div>
                <div className="rounded-3xl border border-white/8 bg-white/5 p-5 backdrop-blur-sm">
                  <div className="text-2xl font-semibold text-white">No hardware swap</div>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    capture ROI on the GPUs and clusters you already run today.
                  </p>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 rounded-[2rem] bg-cyan-400/10 blur-3xl" />
              <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/75 p-5 shadow-[0_30px_120px_rgba(2,6,23,0.9)] backdrop-blur-xl">
                <div className="flex items-center justify-between border-b border-white/8 pb-4">
                  <div>
                    <p className="text-sm font-medium text-white">
                      Fleet optimization control plane
                    </p>
                    <p className="mt-1 text-xs uppercase tracking-[0.24em] text-slate-400">
                      profiler loop / policy learning / rollout safety
                    </p>
                  </div>
                  <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300">
                    Live
                  </div>
                </div>

                <div className="mt-5 grid gap-5 xl:gap-6 lg:grid-cols-[1.3fr_0.9fr]">
                  <div className="rounded-[1.5rem] border border-white/8 bg-slate-900/70 p-4">
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-slate-200">Profiler trace</p>
                      <p className="text-xs text-slate-400">A100 cluster / us-west</p>
                    </div>
                    <div className="mt-5 space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span>Kernel occupancy</span>
                          <span>62% to 84%</span>
                        </div>
                        <div className="h-3 rounded-full bg-white/6 p-0.5">
                          <div className="h-full w-[84%] rounded-full bg-[linear-gradient(90deg,#06b6d4,#8b5cf6)]" />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span>Memory efficiency</span>
                          <span>71% to 89%</span>
                        </div>
                        <div className="h-3 rounded-full bg-white/6 p-0.5">
                          <div className="h-full w-[89%] rounded-full bg-[linear-gradient(90deg,#14b8a6,#22c55e)]" />
                        </div>
                      </div>
                      <div className="rounded-2xl border border-white/8 bg-black/30 p-4">
                        <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-slate-500">
                          <span>Kernel timeline</span>
                          <span>120 ms window</span>
                        </div>
                        <div className="mt-4 space-y-3">
                          {[
                            ["embed_forward", "w-[76%]", "bg-cyan-400"],
                            ["fused_attention", "w-[92%]", "bg-violet-400"],
                            ["kv_cache_repack", "w-[41%]", "bg-teal-400"],
                            ["decode_step", "w-[68%]", "bg-emerald-400"],
                          ].map(([label, width, color]) => (
                            <div key={label} className="grid grid-cols-[110px_1fr] items-center gap-3">
                              <span className="text-xs text-slate-400">{label}</span>
                              <div className="h-4 rounded-full bg-white/6 p-0.5">
                                <div className={`h-full rounded-full ${width} ${color}`} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="rounded-[1.5rem] border border-white/8 bg-white/5 p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                        Suggested actions
                      </p>
                      <div className="mt-4 space-y-3">
                        {[
                          "Fuse attention epilogue to reduce global memory traffic",
                          "Adjust thread block size from 128 to 256 for occupancy gain",
                          "Re-layout cache writes to improve coalescing",
                        ].map((item) => (
                          <div
                            key={item}
                            className="rounded-2xl border border-white/8 bg-slate-950/70 p-3 text-sm leading-6 text-slate-200"
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.5rem] border border-emerald-400/20 bg-emerald-400/10 p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-emerald-200/80">
                        Estimated uplift
                      </p>
                      <div className="mt-4 grid grid-cols-2 gap-3">
                        <div className="rounded-2xl bg-black/20 p-4">
                          <div className="text-2xl font-semibold text-white">-28%</div>
                          <p className="mt-1 text-xs text-emerald-100/80">
                            GPU cost per job
                          </p>
                        </div>
                        <div className="rounded-2xl bg-black/20 p-4">
                          <div className="text-2xl font-semibold text-white">+34%</div>
                          <p className="mt-1 text-xs text-emerald-100/80">
                            Throughput gain
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[1.5rem] border border-white/8 bg-slate-900/70 p-4">
                      <p className="text-sm text-slate-200">Optimization loop</p>
                      <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-300">
                        {["Trace", "Diagnose", "Tune", "Validate", "Deploy", "Learn"].map(
                          (item, index) => (
                            <span
                              key={item}
                              className={`rounded-full px-3 py-2 ${
                                index === 2 || index === 4
                                  ? "border border-cyan-400/30 bg-cyan-400/10 text-cyan-200"
                                  : "border border-white/8 bg-white/5"
                              }`}
                            >
                              {item}
                            </span>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="border-y border-white/8 py-6">
          <div className="grid gap-4 lg:grid-cols-[240px_1fr] lg:items-center">
            <p className="text-xs font-medium uppercase tracking-[0.32em] text-slate-500">
              Built for high-performance teams
            </p>
            <div className="grid grid-cols-2 gap-3 text-sm text-slate-300 sm:grid-cols-3 lg:grid-cols-6">
              {trustCategories.map((item) => (
                <div
                  key={item}
                  className="rounded-full border border-white/8 bg-white/4 px-4 py-3 text-center"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-24 sm:py-28">
          <SectionHeading
            eyebrow="The problem"
            title="GPU performance tuning is still too manual for modern infrastructure."
            description="As GPU fleets grow, the economics get harsher. Teams can see utilization gaps, kernel stalls, and memory waste, but turning those traces into improvements remains slow, fragile, and expensive."
          />
          <div className="mt-12 grid gap-6 lg:grid-cols-3">
            {problemPoints.map((item) => (
              <article
                key={item.title}
                className="rounded-[1.75rem] border border-white/8 bg-white/5 p-7"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 text-sm font-semibold text-cyan-200">
                  0{problemPoints.indexOf(item) + 1}
                </div>
                <h3 className="mt-5 text-xl font-semibold text-white">
                  {item.title}
                </h3>
                <p className="mt-3 text-base leading-7 text-slate-300">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section id="how-it-works" className="py-24 sm:py-28">
          <SectionHeading
            eyebrow="How it works"
            title="A closed-loop system for profiling, optimization, and continuous learning."
            description="GPU Performance Autopilot does more than monitor. It connects production telemetry to optimization policies and safe rollout controls so improvements can ship continuously."
          />
          <div className="mt-12 grid gap-6 lg:grid-cols-2">
            {workflowSteps.map((item) => (
              <article
                key={item.step}
                className="rounded-[1.75rem] border border-white/8 bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(15,23,42,0.55))] p-7"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold tracking-[0.25em] text-cyan-300/80">
                    {item.step}
                  </span>
                  <span className="h-px flex-1 bg-white/8 ml-4" />
                </div>
                <h3 className="mt-5 text-2xl font-semibold tracking-[-0.03em] text-white">
                  {item.title}
                </h3>
                <p className="mt-3 text-base leading-7 text-slate-300">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section id="capabilities" className="py-24 sm:py-28">
          <SectionHeading
            eyebrow="Capabilities"
            title="Built for platform engineers operating real GPU workloads."
            description="The product surface is designed for technical teams that care about telemetry fidelity, optimization quality, and controlled deployment into production systems."
          />
          <div className="mt-12 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {capabilityCards.map((item) => (
              <article
                key={item.title}
                className="group rounded-[1.5rem] border border-white/8 bg-white/5 p-6 transition hover:border-cyan-400/20 hover:bg-white/[0.07]"
              >
                <div className="h-10 w-10 rounded-2xl border border-white/10 bg-slate-900/80" />
                <h3 className="mt-5 text-lg font-semibold text-white">
                  {item.title}
                </h3>
                <p className="mt-3 text-sm leading-7 text-slate-300">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section id="roi" className="py-24 sm:py-28">
          <div className="rounded-[2rem] border border-white/8 bg-[linear-gradient(135deg,rgba(8,15,31,0.94),rgba(15,23,42,0.8))] p-8 sm:p-10 lg:p-12">
            <SectionHeading
              eyebrow="ROI"
              title="Immediate infrastructure leverage, not a long science project."
              description="The business case is direct: reduce wasted GPU spend, increase throughput, and shrink the manual effort required to keep performance healthy as workloads evolve."
            />
            <div className="mt-12 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
              {metrics.map((item) => (
                <div
                  key={item.label}
                  className="rounded-[1.5rem] border border-white/8 bg-black/20 p-6"
                >
                  <div className="text-4xl font-semibold tracking-[-0.05em] text-white">
                    {item.value}
                  </div>
                  <p className="mt-3 text-sm uppercase tracking-[0.2em] text-slate-400">
                    {item.label}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="use-cases" className="py-24 sm:py-28">
          <SectionHeading
            eyebrow="Use cases"
            title="One platform for every team where GPU efficiency matters."
            description="Whether the bottleneck sits in training throughput, inference latency, simulation jobs, or execution engines, the optimization loop stays the same: measure, tune, validate, and keep learning."
          />
          <div className="mt-12 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {useCases.map((item) => (
              <article
                key={item}
                className="rounded-[1.5rem] border border-white/8 bg-white/5 p-6"
              >
                <div className="text-sm uppercase tracking-[0.24em] text-cyan-300/80">
                  {item}
                </div>
                <p className="mt-4 text-base leading-7 text-slate-300">
                  Optimize kernels, memory behavior, and launch patterns across
                  production GPU jobs without relying on a manual tuning backlog.
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="py-24 sm:py-28">
          <div className="grid gap-10 lg:grid-cols-[0.95fr_1.05fr]">
            <div>
              <SectionHeading
                eyebrow="Moat"
                title="A data flywheel that gets smarter as more workloads run through it."
                description="This is not a static rules engine. The product compounds because every analyzed workload expands the space of bottlenecks, optimization outcomes, and policy quality."
              />
            </div>
            <div className="rounded-[2rem] border border-white/8 bg-white/5 p-7">
              <div className="grid gap-4">
                {moatSteps.map((item, index) => (
                  <div key={item} className="flex items-center gap-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 text-sm font-semibold text-cyan-200">
                      0{index + 1}
                    </div>
                    <div className="flex-1 rounded-2xl border border-white/8 bg-slate-950/70 px-4 py-4 text-sm text-slate-200">
                      {item}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6 rounded-[1.5rem] border border-emerald-400/15 bg-emerald-400/10 p-5">
                <p className="text-sm leading-7 text-emerald-50/90">
                  More coverage leads to better optimization policies. Better
                  policies drive better recommendations and safer automation.
                  Better outcomes attract more workloads, strengthening the
                  product again.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 sm:py-16">
          <div className="grid gap-6 lg:grid-cols-2">
            {testimonials.map((item) => (
              <blockquote
                key={item.author}
                className="rounded-[1.75rem] border border-white/8 bg-white/5 p-7"
              >
                <p className="text-lg leading-8 text-slate-100">
                  &ldquo;{item.quote}&rdquo;
                </p>
                <footer className="mt-6 text-sm text-slate-400">
                  {item.author} / {item.team}
                </footer>
              </blockquote>
            ))}
          </div>
        </section>

        <section id="demo" className="py-24 sm:py-28">
          <div className="rounded-[2rem] border border-cyan-400/20 bg-[linear-gradient(135deg,rgba(6,182,212,0.16),rgba(14,165,233,0.08),rgba(15,23,42,0.88))] p-8 sm:p-10 lg:p-14">
            <div className="grid gap-10 lg:grid-cols-[1fr_0.75fr] lg:items-end">
              <div className="max-w-3xl">
                <span className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-200">
                  Final CTA
                </span>
                <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
                  Turn wasted GPU compute into measurable performance gains.
                </h2>
                <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-200">
                  Show your team where GPU efficiency is leaking today, what can
                  be tuned automatically, and how quickly those gains can land in
                  production.
                </p>
              </div>
              <form className="rounded-[1.75rem] border border-white/10 bg-slate-950/75 p-6 backdrop-blur">
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
                    placeholder="Inference, training, simulation..."
                    className="mt-3 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40 focus:bg-white/[0.07]"
                  />
                </label>
                <button
                  type="submit"
                  className="mt-5 inline-flex w-full items-center justify-center rounded-full bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
                >
                  Request early access
                </button>
                <p className="mt-4 text-xs leading-6 text-slate-500">
                  Placeholder form for design purposes. Replace with your CRM,
                  waitlist, or booking flow.
                </p>
              </form>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
