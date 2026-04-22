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
} from "lucide-react";

export const metadata: Metadata = {
  title: "How It Works",
  description:
    "See how Fournex profiles GPU workloads, identifies bottlenecks, ranks fixes, and validates the highest-ROI optimizations.",
};

const workflowSteps = [
  {
    step: "01",
    title: "Capture low-overhead traces",
    description:
      "We collect the minimum profiler, runtime, and allocator signals needed to understand where your GPU time and memory are actually going.",
    bullets: [
      "Kernel, stream, and memory telemetry",
      "Training and inference compatible",
      "Built to run against production workloads",
    ],
    icon: Microscope,
  },
  {
    step: "02",
    title: "Classify the bottleneck",
    description:
      "Instead of dumping raw timelines on your team, we map trace signatures to known bottleneck families and attach a concrete diagnosis.",
    bullets: [
      "Dataloader starvation",
      "Launch fragmentation",
      "Mixed-precision and memory issues",
    ],
    icon: Target,
  },
  {
    step: "03",
    title: "Rank the highest-ROI fixes",
    description:
      "Recommendations are ordered by expected uplift, implementation effort, confidence, and risk so the first action is obvious.",
    bullets: [
      "Explainable ranking logic",
      "Fast wins separated from riskier changes",
      "Optimizations scoped to your workload shape",
    ],
    icon: Sparkles,
  },
  {
    step: "04",
    title: "Validate and close the loop",
    description:
      "Every suggested change is meant to be benchmarked, checked for regressions, and fed back into the policy layer over time.",
    bullets: [
      "Before/after measurement",
      "Guardrails for NaNs, memory, and throughput",
      "Continuous learning from validated outcomes",
    ],
    icon: Workflow,
  },
];

const systemCards = [
  {
    title: "Profiler with opinions",
    description:
      "The system is not just a tracer. It is opinionated about what signals matter and what failure modes are common in real GPU workloads.",
    icon: Gauge,
  },
  {
    title: "Recommendation engine",
    description:
      "Each diagnosis maps to concrete next actions, not generic advice. The goal is to shorten the path from trace to measurable gain.",
    icon: Sparkles,
  },
  {
    title: "Safety layer",
    description:
      "Suggested changes are bounded by confidence, blast radius, and reproducibility so teams can trust what gets proposed.",
    icon: ShieldCheck,
  },
];

const validationChecks = [
  "Throughput improves versus baseline",
  "Memory pressure stays within guardrails",
  "Numerics remain stable",
  "Regression checks pass before rollout",
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
    <div className="max-w-3xl">
      <div className="text-xs font-semibold uppercase tracking-[0.28em] text-violet-300/80">
        {eyebrow}
      </div>
      <h2 className="mt-4 text-3xl font-semibold tracking-[-0.05em] text-white sm:text-4xl lg:text-[2.8rem] lg:leading-[1.05]">
        {title}
      </h2>
      <p className="mt-4 text-base leading-8 text-slate-300 sm:text-lg">
        {description}
      </p>
    </div>
  );
}

export default function HowItWorksPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(109,40,217,0.22),_transparent_22%),radial-gradient(circle_at_85%_20%,_rgba(56,189,248,0.12),_transparent_18%),linear-gradient(180deg,_#03040b_0%,_#070815_38%,_#04050d_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:120px_120px] [mask-image:radial-gradient(circle_at_top,black,transparent_82%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top,_rgba(124,58,237,0.22),_transparent_58%)] blur-3xl" />

      <div className="relative mx-auto flex w-full max-w-[92rem] flex-col px-6 pb-24 pt-10 sm:px-8 lg:px-10 xl:px-12">
        <section className="grid gap-12 border-b border-white/8 pb-16 pt-10 lg:grid-cols-[0.95fr_1.05fr] lg:items-end">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-2 text-[0.78rem] font-semibold text-violet-200">
              <Sparkles size={14} className="text-violet-300" />
              Product walkthrough
            </div>
            <h1 className="mt-6 text-5xl font-semibold tracking-[-0.07em] text-white sm:text-6xl lg:text-[5rem] lg:leading-[0.95]">
              How Fournex
              <br />
              turns traces into
              <span className="text-violet-400"> validated speedups.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300">
              The workflow is simple: profile a real GPU workload, diagnose the
              real bottleneck, rank the fixes that matter, and validate the
              change before it lands in production.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/#demo"
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-6 py-4 text-sm font-semibold text-white transition hover:from-violet-500 hover:to-fuchsia-400"
              >
                Request access
                <ArrowRight size={16} />
              </Link>
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/12 bg-white/[0.03] px-6 py-4 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/[0.06]"
              >
                View analysis flow
              </Link>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-6 shadow-[0_35px_100px_rgba(3,6,18,0.45)]">
            <div className="grid gap-4 sm:grid-cols-3">
              {systemCards.map((card) => {
                const Icon = card.icon;
                return (
                  <article
                    key={card.title}
                    className="rounded-[1.4rem] border border-white/8 bg-[#090c18] p-5"
                  >
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-violet-300">
                      <Icon size={18} />
                    </div>
                    <h3 className="mt-5 text-lg font-semibold text-white">
                      {card.title}
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-slate-300">
                      {card.description}
                    </p>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        <section className="py-20 sm:py-24">
          <SectionHeading
            eyebrow="Step By Step"
            title="A four-stage loop designed for production GPU teams."
            description="The important part is not just collecting data. The product is the system that converts noisy runtime traces into ordered, explainable, and testable actions."
          />

          <div className="mt-12 grid gap-5 xl:grid-cols-2">
            {workflowSteps.map((item) => {
              const Icon = item.icon;
              return (
                <article
                  key={item.step}
                  className="rounded-[1.7rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.02))] p-6"
                >
                  <div className="flex items-start justify-between gap-6">
                    <div className="max-w-xl">
                      <div className="text-[0.72rem] font-semibold uppercase tracking-[0.24em] text-violet-300/80">
                        Step {item.step}
                      </div>
                      <h3 className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">
                        {item.title}
                      </h3>
                      <p className="mt-4 text-sm leading-7 text-slate-300 sm:text-base">
                        {item.description}
                      </p>
                    </div>
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 text-violet-300">
                      <Icon size={20} />
                    </div>
                  </div>

                  <div className="mt-6 grid gap-3 text-sm text-slate-300">
                    {item.bullets.map((bullet) => (
                      <div
                        key={bullet}
                        className="flex items-start gap-3 rounded-2xl border border-white/8 bg-[#090c18] px-4 py-3"
                      >
                        <CheckCircle2
                          size={16}
                          className="mt-0.5 shrink-0 text-emerald-300"
                        />
                        <span>{bullet}</span>
                      </div>
                    ))}
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="grid gap-8 border-y border-white/8 py-20 sm:py-24 lg:grid-cols-[1.05fr_0.95fr] lg:items-start">
          <div>
            <SectionHeading
              eyebrow="Validation"
              title="Recommendations are only useful if teams can trust them."
              description="A good optimization system cannot stop at suggestions. It needs guardrails, confidence signals, and a way to prove that the proposed change actually improved the workload."
            />
            <div className="mt-8 grid gap-3 text-sm text-slate-300">
              {validationChecks.map((item) => (
                <div
                  key={item}
                  className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3"
                >
                  <ShieldCheck
                    size={16}
                    className="mt-0.5 shrink-0 text-cyan-300"
                  />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-[#090c18] p-6">
            <div className="text-sm font-medium text-slate-300">
              Example operating model
            </div>
            <div className="mt-6 space-y-4">
              {[
                {
                  label: "Input",
                  value:
                    "Live workload traces from training, inference, or simulation jobs",
                },
                {
                  label: "Reasoning",
                  value:
                    "Classifier maps telemetry patterns to bottlenecks and possible fixes",
                },
                {
                  label: "Decision",
                  value:
                    "System ranks the next action by ROI, effort, confidence, and risk",
                },
                {
                  label: "Output",
                  value:
                    "Teams get a validated optimization path instead of another dashboard",
                },
              ].map((row) => (
                <div
                  key={row.label}
                  className="rounded-[1.2rem] border border-white/8 bg-white/[0.03] px-4 py-4"
                >
                  <div className="text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-violet-300/80">
                    {row.label}
                  </div>
                  <p className="mt-2 text-sm leading-7 text-slate-300">
                    {row.value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-20 sm:py-24">
          <div className="overflow-hidden rounded-[2rem] border border-violet-400/20 bg-[linear-gradient(135deg,rgba(124,58,237,0.16),rgba(20,24,40,0.96))] p-8 sm:p-10 lg:p-12">
            <div className="max-w-3xl">
              <div className="text-xs font-semibold uppercase tracking-[0.28em] text-violet-200/80">
                Next step
              </div>
              <h2 className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
                See what your first validated optimization report looks like.
              </h2>
              <p className="mt-5 max-w-2xl text-base leading-8 text-slate-200 sm:text-lg">
                If the workflow makes sense, the next question is simple: what
                would the system find in your actual GPU workload?
              </p>
              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <Link
                  href="/#demo"
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-white px-6 py-4 text-sm font-semibold text-slate-950 transition hover:bg-slate-100"
                >
                  Request early access
                  <ArrowRight size={16} />
                </Link>
                <Link
                  href="/"
                  className="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/12 bg-white/[0.03] px-6 py-4 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/[0.06]"
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
