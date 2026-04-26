import type { Metadata } from "next";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Gauge,
  Microscope,
  ShieldCheck,
  Sparkles,
  Target,
  Workflow,
  Activity,
  Cpu,
  LineChart,
} from "lucide-react";

export const metadata: Metadata = {
  title: "How It Works | Fournex",
  description:
    "See how Fournex profiles GPU workloads, identifies bottlenecks, ranks fixes, and validates the highest-ROI optimizations.",
};

const workflowSteps =[
  {
    step: "1",
    title: "Capture low-overhead traces",
    description:
      "We collect the minimum profiler, runtime, and allocator signals needed to understand where your GPU time and memory are actually going.",
    bullets:[
      "Kernel, stream, and memory telemetry",
      "Training and inference compatible",
      "Built to run against production workloads",
    ],
    icon: Microscope,
  },
  {
    step: "2",
    title: "Classify the bottleneck",
    description:
      "Instead of dumping raw timelines on your team, we map trace signatures to known bottleneck families and attach a concrete diagnosis.",
    bullets:[
      "Dataloader starvation",
      "Launch fragmentation",
      "Mixed-precision and memory issues",
    ],
    icon: Target,
  },
  {
    step: "3",
    title: "Rank the highest-ROI fixes",
    description:
      "Recommendations are ordered by expected uplift, implementation effort, confidence, and risk so the first action is obvious.",
    bullets:[
      "Explainable ranking logic",
      "Fast wins separated from riskier changes",
      "Optimizations scoped to your workload shape",
    ],
    icon: Sparkles,
  },
  {
    step: "4",
    title: "Validate and close the loop",
    description:
      "Every suggested change is meant to be benchmarked, checked for regressions, and fed back into the policy layer over time.",
    bullets:[
      "Before/after measurement",
      "Guardrails for NaNs, memory, and throughput",
      "Continuous learning from validated outcomes",
    ],
    icon: Workflow,
  },
];

const systemCards =[
  {
    title: "Profiler with opinions",
    description:
      "The system is opinionated about what signals matter and what failure modes are common in real GPU workloads.",
    icon: Gauge,
  },
  {
    title: "Recommendation engine",
    description:
      "Each diagnosis maps to concrete next actions, shortening the path from raw trace to measurable gain.",
    icon: Cpu,
  },
  {
    title: "Safety layer",
    description:
      "Suggested changes are bounded by confidence, blast radius, and reproducibility so teams can trust the proposals.",
    icon: ShieldCheck,
  },
];

const validationChecks =[
  "Throughput improves versus baseline",
  "Memory pressure stays within guardrails",
  "Numerics remain stable",
  "Regression checks pass before rollout",
];

const operatingModel =[
  {
    label: "Input",
    value: "Live workload traces from training, inference, or simulation jobs",
    icon: Activity,
  },
  {
    label: "Reasoning",
    value: "Classifier maps telemetry patterns to bottlenecks and possible fixes",
    icon: Cpu,
  },
  {
    label: "Decision",
    value: "System ranks the next action by ROI, effort, confidence, and risk",
    icon: Target,
  },
  {
    label: "Output",
    value: "Teams get a validated optimization path instead of a messy dashboard",
    icon: LineChart,
  },
];

function SectionHeading({
  eyebrow,
  title,
  description,
  centered = false,
}: {
  eyebrow: string;
  title: string;
  description: string;
  centered?: boolean;
}) {
  return (
    <div className={`max-w-3xl ${centered ? "mx-auto text-center" : ""}`}>
      <div className="text-xs font-bold uppercase tracking-[0.28em] text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">
        {eyebrow}
      </div>
      <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-4xl lg:text-5xl lg:leading-[1.1]">
        {title}
      </h2>
      <p className="mt-4 text-base leading-relaxed text-slate-400 sm:text-lg">
        {description}
      </p>
    </div>
  );
}

export default function HowItWorksPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[#03040b] text-white selection:bg-violet-500/30">
      {/* Background Effects */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(109,40,217,0.15),_transparent_25%),radial-gradient(circle_at_85%_30%,_rgba(56,189,248,0.08),_transparent_20%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px][mask-image:radial-gradient(circle_at_top,black,transparent_80%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[500px] bg-[radial-gradient(ellipse_at_top,_rgba(124,58,237,0.15),_transparent_50%)] blur-3xl" />

      <div className="relative mx-auto flex w-full max-w-[88rem] flex-col px-5 pb-24 pt-12 sm:px-8 lg:px-12">
        
        {/* --- Hero Section --- */}

        {/* --- Workflow Steps --- */}
        <section className="py-24 sm:py-32">
          <SectionHeading
            eyebrow="Step By Step"
            title="A four-stage loop designed for production GPU teams."
            description="The important part is not just collecting data. The product is the system that converts noisy runtime traces into ordered, explainable, and testable actions."
            centered
          />

          <div className="mt-16 grid gap-6 md:grid-cols-2">
            {workflowSteps.map((item) => {
              const Icon = item.icon;
              return (
                <article
                  key={item.step}
                  className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.02] p-8 transition-all hover:border-violet-500/30 hover:bg-white/[0.04] sm:p-10"
                >
                  {/* Big background number */}
                  <div className="pointer-events-none absolute -bottom-6 -right-4 text-[10rem] font-bold leading-none text-white/[0.02] transition-colors group-hover:text-violet-500/5 select-none">
                    0{item.step}
                  </div>

                  <div className="relative z-10 flex items-start justify-between gap-6">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-300 transition-colors group-hover:border-violet-400/30 group-hover:bg-violet-500/20 group-hover:text-violet-300">
                      <Icon size={22} />
                    </div>
                    <div className="mt-2 text-xs font-bold uppercase tracking-[0.2em] text-violet-400/80">
                      Step {item.step}
                    </div>
                  </div>

                  <div className="relative z-10 mt-8 max-w-xl">
                    <h3 className="text-2xl font-semibold tracking-tight text-white">
                      {item.title}
                    </h3>
                    <p className="mt-4 text-base leading-relaxed text-slate-400">
                      {item.description}
                    </p>
                  </div>

                  <div className="relative z-10 mt-8 grid gap-3 text-sm text-slate-300">
                    {item.bullets.map((bullet) => (
                      <div
                        key={bullet}
                        className="flex items-start gap-3 rounded-xl border border-white/5 bg-black/20 px-4 py-3 backdrop-blur-sm transition-colors group-hover:border-white/10"
                      >
                        <CheckCircle2
                          size={18}
                          className="mt-0.5 shrink-0 text-emerald-400/80"
                        />
                        <span className="leading-snug">{bullet}</span>
                      </div>
                    ))}
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {/* --- Validation & Operating Model --- */}
        <section className="grid gap-16 border-y border-white/10 py-24 sm:py-32 lg:grid-cols-2 lg:items-center">
          <div>
            <SectionHeading
              eyebrow="Validation"
              title="Recommendations are only useful if teams can trust them."
              description="A good optimization system cannot stop at suggestions. It needs guardrails, confidence signals, and a way to prove that the proposed change actually improved the workload."
            />
            <div className="mt-10 grid gap-4 text-sm text-slate-300 sm:text-base">
              {validationChecks.map((item) => (
                <div
                  key={item}
                  className="group flex items-center gap-4 rounded-2xl border border-white/5 bg-white/[0.02] px-5 py-4 transition-colors hover:border-cyan-500/30 hover:bg-white/[0.04]"
                >
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-cyan-500/10 text-cyan-400">
                    <ShieldCheck size={20} />
                  </div>
                  <span className="font-medium">{item}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="relative rounded-[2rem] border border-white/10 bg-[#090c18] p-8 shadow-2xl sm:p-10">
            <div className="mb-8 text-sm font-semibold tracking-wide text-slate-400 uppercase">
              Example Operating Pipeline
            </div>
            
            {/* Connected Pipeline Visual */}
            <div className="relative space-y-6 before:absolute before:inset-y-6 before:left-[1.65rem] before:w-[2px] before:bg-gradient-to-b before:from-violet-500/50 before:via-white/10 before:to-transparent">
              {operatingModel.map((row, i) => {
                const Icon = row.icon;
                return (
                  <div key={row.label} className="relative flex items-start gap-5 group">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full border-4 border-[#090c18] bg-white/[0.05] text-slate-400 z-10 transition-colors group-hover:bg-violet-500/20 group-hover:text-violet-300 group-hover:border-violet-500/30">
                      <Icon size={20} />
                    </div>
                    <div className="flex-1 rounded-[1.2rem] border border-white/5 bg-white/[0.02] p-5 transition-colors group-hover:border-white/10 group-hover:bg-white/[0.04]">
                      <div className="text-[0.7rem] font-bold uppercase tracking-[0.2em] text-violet-400">
                        {row.label}
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-slate-300">
                        {row.value}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* --- CTA Section --- */}
        <section className="py-24 sm:py-32">
          <div className="relative overflow-hidden rounded-[2.5rem] border border-violet-500/20 bg-black p-8 sm:p-12 lg:p-16">
            {/* Gradient meshes for CTA background */}
            <div className="pointer-events-none absolute -right-20 -top-20 h-[400px] w-[400px] rounded-full bg-violet-600/20 blur-[100px]" />
            <div className="pointer-events-none absolute -bottom-20 -left-20 h-[400px] w-[400px] rounded-full bg-cyan-600/10 blur-[100px]" />
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20" />

            <div className="relative z-10 max-w-3xl">
              <div className="text-xs font-bold uppercase tracking-[0.25em] text-violet-300">
                Next step
              </div>
              <h2 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-5xl lg:leading-[1.1]">
                See what your first validated optimization report looks like.
              </h2>
              <p className="mt-6 max-w-2xl text-base leading-relaxed text-slate-300 sm:text-lg">
                If the workflow makes sense, the next question is simple: what
                would the system find in your actual GPU workload?
              </p>
              <div className="mt-10 flex flex-col gap-4 sm:flex-row">
                <Link
                  href="/#demo"
                  className="group inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-8 py-4 text-sm font-bold text-slate-950 transition-all hover:scale-[1.02] hover:bg-slate-100 sm:w-auto"
                >
                  Request early access
                  <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
                </Link>
                <Link
                  href="/"
                  className="inline-flex w-full items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-8 py-4 text-sm font-semibold text-white backdrop-blur-sm transition-all hover:bg-white/[0.08] sm:w-auto"
                >
                  Back to homepage
                </Link>
              </div>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}