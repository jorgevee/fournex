import type { Metadata } from "next";
import Link from "next/link";
import {
  Terminal,
  FolderOpen,
  Stethoscope,
  FlaskConical,
  BarChart3,
  ChevronRight,
  Package,
  Cpu,
  Zap,
  SlidersHorizontal,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Docs - Fournex",
  description: "Developer reference for the frx CLI.",
};

function Section({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 space-y-4">
      {children}
    </section>
  );
}

function H2({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="border-b border-white/10 pb-3 text-xl font-semibold tracking-tight text-white">
      {children}
    </h2>
  );
}

function H3({ children }: { children: React.ReactNode }) {
  return <h3 className="text-base font-semibold text-slate-200">{children}</h3>;
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-sm leading-7 text-slate-400">{children}</p>;
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[0.78rem] text-sky-300">
      {children}
    </code>
  );
}

function Pre({ children }: { children: React.ReactNode }) {
  return (
    <pre className="overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 font-mono text-[0.78rem] leading-6 text-slate-300">
      {children}
    </pre>
  );
}

function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full text-left text-sm">{children}</table>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="border-b border-white/10 bg-white/[0.04] px-4 py-3 font-semibold text-slate-300">
      {children}
    </th>
  );
}

function Td({ children, mono }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <td className={`border-b border-white/[0.06] px-4 py-3 text-slate-400 ${mono ? "font-mono text-xs text-sky-300" : ""}`}>
      {children}
    </td>
  );
}

function NavItem({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <li>
      <a
        href={href}
        className="flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm text-slate-400 transition hover:bg-white/[0.06] hover:text-slate-200"
      >
        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-slate-600" />
        {children}
      </a>
    </li>
  );
}

function CommandBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-0.5 font-mono text-[0.7rem] text-sky-300">
      {children}
    </span>
  );
}

export default function DocsPage() {
  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,_#03040b_0%,_#070815_38%,_#04050d_100%)] text-white">
      <div className="pointer-events-none fixed inset-0 bg-[linear-gradient(rgba(148,163,184,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.03)_1px,transparent_1px)] bg-[size:80px_80px] [mask-image:radial-gradient(circle_at_top,black,transparent_75%)]" />

      <div className="relative mx-auto max-w-[88rem] px-6 py-12 sm:px-8 lg:px-10">

        {/* Header */}
        <div className="mb-12 border-b border-white/10 pb-8">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-4">
            <Link href="/" className="hover:text-slate-300 transition">Home</Link>
            <ChevronRight className="h-3 w-3" />
            <span className="text-slate-400">Docs</span>
          </div>
          <div className="flex items-center gap-3 mb-3">
            <Terminal className="h-7 w-7 text-violet-400" />
            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Fournex — CLI Reference
            </h1>
          </div>
          <p className="text-slate-400 text-sm max-w-2xl">
            Developer documentation for the <Code>frx</Code> CLI.
            Covers installation, all CLI subcommands, bundle layout, and how
            the analysis pipeline works.
          </p>
        </div>

        <div className="flex gap-10 lg:gap-14">

          {/* Sidebar */}
          <nav className="hidden shrink-0 lg:block" style={{ width: "14rem" }}>
            <div className="sticky top-8 space-y-1">
              <p className="mb-2 px-3 text-[0.65rem] font-semibold uppercase tracking-widest text-slate-600">
                Getting started
              </p>
              <ul className="space-y-0.5">
                <NavItem href="#install">Installation</NavItem>
                <NavItem href="#quickstart">Quickstart</NavItem>
              </ul>
              <p className="mb-2 mt-6 px-3 text-[0.65rem] font-semibold uppercase tracking-widest text-slate-600">
                Commands
              </p>
              <ul className="space-y-0.5">
                <NavItem href="#collect">collect</NavItem>
                <NavItem href="#analyze">analyze</NavItem>
                <NavItem href="#tune">tune</NavItem>
                <NavItem href="#doctor">doctor</NavItem>
                <NavItem href="#smoke-test">smoke-test</NavItem>
              </ul>
              <p className="mb-2 mt-6 px-3 text-[0.65rem] font-semibold uppercase tracking-widest text-slate-600">
                Reference
              </p>
              <ul className="space-y-0.5">
                <NavItem href="#bundle-layout">Bundle layout</NavItem>
                <NavItem href="#analysis">Analysis pipeline</NavItem>
                <NavItem href="#bottlenecks">Bottleneck labels</NavItem>
                <NavItem href="#sdk">SDK integration</NavItem>
              </ul>
            </div>
          </nav>

          {/* Content */}
          <div className="min-w-0 flex-1 space-y-14">

            {/* Installation */}
            <Section id="install">
              <H2>Installation</H2>
              <P>
                Install the package from the repo root. Python 3.11+ is required.
                PyTorch is optional — the CLI works without it for bundle analysis;
                it is only needed when the SDK instruments a live training run.
              </P>
              <Pre>{`pip install fournex`}</Pre>
              <P>
                This registers the <Code>frx</Code> entry point.
                Verify with:
              </P>
              <Pre>{`frx doctor`}</Pre>
            </Section>

            {/* Quickstart */}
            <Section id="quickstart">
              <H2>Quickstart</H2>
              <div className="space-y-4">
                <P>
                  The typical workflow is three steps: collect, inspect locally,
                  then upload to the web analyzer for a full interactive report.
                </P>

                <div className="grid gap-4 sm:grid-cols-3">
                  {[
                    {
                      n: "1",
                      icon: Terminal,
                      title: "Collect",
                      body: "Wrap your training script. The CLI captures GPU metrics, imports profiler artifacts, and generates a pre-analyzed bundle.",
                    },
                    {
                      n: "2",
                      icon: BarChart3,
                      title: "Analyze locally",
                      body: "Print the diagnosis to the terminal without uploading anything.",
                    },
                    {
                      n: "3",
                      icon: Zap,
                      title: "Upload",
                      body: "Drop the zip on the web analyzer for interactive charts, recommendation cards, and shareable links.",
                    },
                  ].map(({ n, icon: Icon, title, body }) => (
                    <div
                      key={n}
                      className="rounded-xl border border-white/10 bg-white/[0.03] p-4"
                    >
                      <div className="mb-3 flex items-center gap-2">
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-500/20 text-xs font-bold text-violet-300">
                          {n}
                        </span>
                        <Icon className="h-4 w-4 text-slate-400" />
                        <span className="text-sm font-semibold text-slate-200">{title}</span>
                      </div>
                      <p className="text-xs leading-5 text-slate-500">{body}</p>
                    </div>
                  ))}
                </div>

                <Pre>{`# 1. Collect
frx collect -- python train.py

# 2. Analyze locally (run-<id> is printed by collect)
frx analyze runs/run-<id>

# 3. Upload  →  drag runs/run-<id>.zip onto fournex.com/analyze

# Optional: let autopilot sweep configs and find the fastest safe candidate
frx tune --safe --max-trials 12 -- python train.py`}</Pre>
              </div>
            </Section>

            {/* collect */}
            <Section id="collect">
              <div className="flex items-center gap-3">
                <H2>collect</H2>
                <CommandBadge>frx collect</CommandBadge>
              </div>

              <P>
                Runs a workload subprocess, samples GPU metrics in the background,
                imports profiler artifacts, runs the analysis pipeline, and writes a
                self-contained run bundle.
              </P>

              <Pre>{`frx collect [OPTIONS] -- COMMAND [ARGS...]

Options:
  --name NAME              Human-readable job name (default: frx-run)
  --out DIR                Root output directory (default: runs)
  --run-id ID              Override auto-generated run ID
  --artifact-dir DIR       Import artifacts from DIR after the workload exits.
                           May be repeated. Default: ./frx-job-run
  --no-profiler-import     Skip importing profiler_trace.json from artifact dirs
  --sample-interval-ms N   nvidia-smi polling interval in ms (default: 1000)
  --config FILE            Optional run_config.yaml to merge into bundle config
  --no-zip                 Skip creating the zip archive`}</Pre>

              <div className="rounded-xl border border-amber-500/20 bg-amber-500/[0.06] p-4">
                <p className="text-sm font-semibold text-amber-300 mb-1">Artifact directory gotcha</p>
                <p className="text-xs leading-5 text-slate-400">
                  If your workload writes <Code>profiler_trace.json</Code> somewhere
                  other than <Code>frx-job-run/</Code>, pass that directory with{" "}
                  <Code>--artifact-dir</Code>. Otherwise the trace exists on disk but
                  will not be copied into the run bundle.
                </p>
              </div>

              <Pre>{`# Workload writes gpu-job-run-tiny-kernels/profiler_trace.json
frx collect \\
  --name tiny-kernel-launch-overhead \\
  --out runs \\
  --sample-interval-ms 100 \\
  --artifact-dir gpu-job-run-tiny-kernels \\
  -- python tiny_kernel_launch_overhead.py --output-dir gpu-job-run-tiny-kernels`}</Pre>

              <H3>What it does</H3>
              <ol className="ml-4 space-y-2 text-sm text-slate-400 list-decimal list-outside">
                <li>Writes <Code>run_config.yaml</Code> and injects env vars into the workload process so the SDK auto-persists events to <Code>raw/trace.jsonl</Code>.</li>
                <li>Starts a background thread that polls <Code>nvidia-smi</Code> at <Code>--sample-interval-ms</Code> into <Code>gpu_metrics.csv</Code>.</li>
                <li>Runs the workload. Stdout and stderr are tee&apos;d to <Code>optional_logs.txt</Code>.</li>
                <li>After the workload exits, copies artifacts from <Code>--artifact-dir</Code> into the bundle (marked <Code>[imported]</Code> in the summary).</li>
                <li>Runs the analysis pipeline over <Code>raw/trace.jsonl</Code> (or the imported profiler bundle if no SDK trace exists) and writes <Code>derived/summary.json</Code>.</li>
                <li>Writes <Code>metadata.json</Code>, <Code>manifest.json</Code>, and zips the bundle.</li>
              </ol>

              <H3>Environment variables injected into the workload</H3>
              <Table>
                <thead>
                  <tr>
                    <Th>Variable</Th>
                    <Th>Value</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["FRX_RUN_ID", "Generated run ID"],
                    ["FRX_JOB_NAME", "--name value"],
                    ["FRX_OUTPUT_DIR", "Absolute path to the run directory"],
                    ["FRX_RAW_TRACE_PATH", "raw/trace.jsonl absolute path"],
                    ["FRX_DERIVED_SUMMARY_PATH", "derived/summary.json absolute path"],
                    ["FRX_AUTO_PERSIST", "1"],
                    ["FRX_SAMPLE_INTERVAL_MS", "--sample-interval-ms value"],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <Td mono>{k}</Td>
                      <Td>{v}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Example output</H3>
              <Pre>{`frx collect completed
Run bundle: runs/run-a1b2c3d4e5f6
Zip bundle: runs/run-a1b2c3d4e5f6.zip

Captured (7 files):
  metadata.json
  manifest.json
  run_config.yaml
  gpu_metrics.csv
  optional_logs.txt
  raw/trace.jsonl
  derived/summary.json
  profiler/profiler_trace.json  [imported]`}</Pre>
            </Section>

            {/* analyze */}
            <Section id="analyze">
              <div className="flex items-center gap-3">
                <H2>analyze</H2>
                <CommandBadge>frx analyze</CommandBadge>
              </div>

              <P>
                Loads a collected run bundle and prints a full diagnosis report to
                stdout. No GPU or PyTorch required.
              </P>

              <Pre>{`frx analyze RUN_DIR [OPTIONS]

Arguments:
  RUN_DIR    Path to a run directory (e.g. runs/run-a1b2c3d4e5f6)
             Must be a directory; zip analysis is not yet supported.

Options:
  --scope    run | steady_state | auto
             Which analysis scope to display.
             auto (default): prefers steady_state when available.
  --json     Output the raw summary JSON instead of the formatted report.`}</Pre>

              <H3>Data source priority</H3>
              <P>
                <Code>analyze</Code> picks the best available data source in this
                order:
              </P>
              <ol className="ml-4 space-y-1 text-sm text-slate-400 list-decimal list-outside">
                <li><Code>derived/summary.json</Code> — pre-analyzed, preferred</li>
                <li><Code>raw/trace.jsonl</Code> — re-analyzed on the fly</li>
                <li><Code>profiler/profiler_trace.json</Code> + <Code>gpu_metrics.csv</Code> — imported and analyzed</li>
              </ol>

              <H3>Example output</H3>
              <Pre>{`--------------------------------------------------------
  GPU Autopilot - Run Analysis
  Run  : run-a1b2c3d4e5f6
  Scope: steady_state  (28 steps)
--------------------------------------------------------

VERDICT
  Primary Bottleneck : input_bound
  Internal Signal    : underutilized_gpu (symptom)
  Confidence         : high (0.88)
  Reason             : input_bound leads the ranking and matches the dominant stall summary.

EVIDENCE
  - Average DataLoader wait fraction is 0.825.
  - Run summary dominant stall type is input_bound.

PERFORMANCE SNAPSHOT
  Avg GPU Utilization : 1.3%
  Avg Memory Util     : 12.0%
  Peak Memory Pressure: 0.14
  Avg Step Time       : 207.000 ms
  Throughput          : 4.8 steps/sec
  Dominant Stall      : input_bound

TOP RECOMMENDATIONS (3 of 5)

  1. [HIGH] Increase DataLoader num_workers
     Effort: low  |  Risk: low  |  Score: 0.84
     DataLoader wait is the dominant stall ...`}</Pre>

              <P>
                When <Code>underutilized_gpu</Code> is the internal top signal but a
                stall type (e.g. <Code>input_bound</Code>) is also present, the
                verdict displays the root cause. The raw internal signal is shown on
                the <Code>Internal Signal</Code> line.
              </P>

              <H3>Launch-bound traces and near-zero GPU samples</H3>
              <P>
                For tiny-kernel workloads, <Code>nvidia-smi</Code> sampling can report
                near-zero GPU utilization even when the profiler captured many CUDA
                kernels. Treat that as bursty GPU activity rather than proof that no
                GPU work ran. The launch-bound report uses profiler evidence such as{" "}
                <Code>kernel_count_per_step</Code>,{" "}
                <Code>median_cuda_kernel_duration_us</Code>,{" "}
                <Code>small_kernel_fraction</Code>, and stable shapes when available.
              </P>
              <Pre>{`VERDICT
  Primary Bottleneck : launch_bound
  Confidence         : medium (0.65)

EVIDENCE
  - Profiler saw about 840.0 CUDA kernels per step with median duration 4.200 us.
  - GPU utilization sampling stayed low, which is expected for bursty tiny-kernel workloads.
  - Shapes were stable, so compile or CUDA graph mitigations are viable.`}</Pre>
            </Section>

            {/* doctor */}
            <Section id="doctor">
              <div className="flex items-center gap-3">
                <H2>doctor</H2>
                <CommandBadge>frx doctor</CommandBadge>
              </div>

              <P>
                Checks that all runtime dependencies are present and configured.
                Exits with code 0 if all checks pass, 1 if any <Code>[FAIL]</Code>{" "}
                lines appear.
              </P>

              <Pre>{`frx doctor`}</Pre>

              <H3>Checks performed</H3>
              <Table>
                <thead>
                  <tr>
                    <Th>Check</Th>
                    <Th>What it verifies</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Python", "Python version (always passes)"],
                    ["torch", "PyTorch importable; reports version"],
                    ["CUDA available", "torch.cuda.is_available(), GPU name and count"],
                    ["nvidia-smi", "nvidia-smi on PATH (required for gpu_metrics.csv)"],
                    ["fournex.profiler", "SDK profiler module importable"],
                    ["fournex.analysis", "Analysis pipeline importable"],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <Td mono>{k}</Td>
                      <Td>{v}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <Pre>{`frx doctor

  [OK]    Python                               3.12.3
  [OK]    torch                                2.3.0+cu121
  [OK]    CUDA available                       NVIDIA A100 x1
  [OK]    nvidia-smi                           /usr/bin/nvidia-smi
  [OK]    fournex.profiler         importable
  [OK]    fournex.analysis         importable

All checks passed.`}</Pre>
            </Section>

            {/* smoke-test */}
            <Section id="smoke-test">
              <div className="flex items-center gap-3">
                <H2>smoke-test</H2>
                <CommandBadge>frx smoke-test</CommandBadge>
              </div>

              <P>
                Writes a synthetic input-bound Chrome-format profiler trace, runs
                the full collect + analysis pipeline end-to-end in a temp directory,
                and verifies the bundle and diagnosis output. Useful for CI and
                confirming the install is working.
              </P>

              <Pre>{`frx smoke-test`}</Pre>

              <P>Checks performed:</P>
              <ul className="ml-4 space-y-1 text-sm text-slate-400 list-disc list-outside">
                <li>Run directory and subdirs exist (<Code>raw/</Code>, <Code>derived/</Code>, <Code>profiler/</Code>)</li>
                <li><Code>derived/summary.json</Code> was generated</li>
                <li><Code>manifest.json</Code> is present</li>
                <li>Zip bundle was created</li>
                <li>Diagnosis produced <Code>primary_bottleneck == input_bound</Code></li>
                <li>At least one recommendation was generated</li>
              </ul>

              <Pre>{`frx smoke-test

Running smoke test ...

  [PASS]  create run directory
  [PASS]  write synthetic profiler trace
  [PASS]  generate derived/summary.json
  [PASS]  manifest.json present
  [PASS]  zip bundle created
  [PASS]  primary_bottleneck == input_bound
  [PASS]  recommendations present
  [PASS]  no unexpected warnings

All smoke-test checks passed.`}</Pre>
            </Section>

            {/* tune */}
            <Section id="tune">
              <div className="flex items-center gap-3">
                <H2>tune</H2>
                <CommandBadge>frx tune</CommandBadge>
                <SlidersHorizontal className="h-4 w-4 text-emerald-400" />
                <span className="inline-flex items-center rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[0.65rem] font-semibold text-emerald-300 uppercase tracking-wide">
                  Autopilot
                </span>
              </div>

              <P>
                Runs the experiment runner: captures a baseline, focuses candidate
                configs from a bottleneck diagnosis, validates safety before each
                trial, measures an explicit benchmark window, rejects quality
                regressions, and writes reproducible artifacts. The command remains
                recommendation-only; it does not rewrite your training config.
              </P>

              <Pre>{`frx tune [OPTIONS] -- COMMAND [ARGS...]

Options:
  --name NAME              Job name for output directories (default: frx-tune)
  --out DIR                Root output directory (default: runs)
  --max-trials N           Max candidate configs to try (default: 12)
  --safe                   Tier-0 only: dataloader knobs (default)
  --no-safe                Also try Tier-1: batch size and mixed precision
  --time-budget-s N        Kill trial after N seconds (default: 60)
  --warmup-steps N         Steps to skip before measuring (default: 5)
  --measure-steps N        Steps to include in measurement (default: 20)
  --repeat-count N         Repeats per baseline and candidate (default: 1)
  --no-race                Disable quick candidate screening
  --race-promote-count N   Candidates promoted from race to full benchmark (default: 3)
  --race-warmup-steps N    Warmup steps for quick screening (default: 1)
  --race-measure-steps N   Measurement steps for quick screening (default: 5)
  --bottleneck LABEL       Focus candidates manually
  --min-speedup FLOAT      Minimum improvement to recommend (default: 0.08 = 8%)
  --allow-risky-actions    Allow high-risk candidates
  --no-quality-checks      Do not require quality checks for precision changes
  --max-final-loss-regression FLOAT
  --max-loss-divergence FLOAT
  --output-abs-tolerance FLOAT
  --allow-nonfinite-loss
  --sample-interval-ms N   GPU sampling interval (default: 1000)`}</Pre>

              <H3>Safety tiers</H3>
              <Table>
                <thead>
                  <tr>
                    <Th>Tier</Th>
                    <Th>Actions</Th>
                    <Th>Flag</Th>
                    <Th>Guardrails</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["0 — Safe", "num_workers, pin_memory, prefetch_factor, persistent_workers", "--safe (default)", "Exit code, step count, throughput not zero"],
                    ["1 — Validated", "batch_size, AMP fp16/bf16", "--no-safe", "Same as Tier 0 + memory ratio < 90%, step time regression < 10%"],
                    ["2 — Risky", "distributed tuning, custom kernels", "Not yet implemented", "Requires explicit user approval"],
                  ].map(([tier, actions, flag, guards]) => (
                    <tr key={tier}>
                      <Td mono>{tier}</Td>
                      <Td>{actions}</Td>
                      <Td mono>{flag}</Td>
                      <Td>{guards}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <P>
                Current implementation adds allocator candidates in the safe tier
                and runtime candidates such as <Code>torch.compile</Code> and CUDA
                Graphs in the validated tier when their preconditions pass.
              </P>

              <H3>Staged search order</H3>
              <P>
                Candidates are generated in stages so the trial budget is spent
                efficiently — no brute-force grid across all knob combinations.
              </P>
              <Pre>{`Screen   race pass    short benchmark all candidates, then promote top N
Stage 1  dataloader   num_workers × pin_memory grid + prefetch_factor variants
Stage 2  batch size   1.25×, 1.5×, 2× baseline  (--no-safe required)
Stage 3  precision    bf16 (Ampere+), fp16        (--no-safe required)`}</Pre>

              <P>
                Race-stage trials are screening signals only. The final winner
                must still come from a full benchmark and pass the normal guard,
                quality, and noise checks.
              </P>

              <H3>Recommendations vs. tune trials</H3>
              <P>
                Recommendations are diagnosis-driven fix cards. They are ranked
                by signal strength, expected impact, effort, and risk, but they
                are not proof that a change already improved your workload.
                Tune trials are executable config candidates that the runner
                actually benchmarks.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Surface</Th>
                    <Th>Source</Th>
                    <Th>Use it for</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Recommendation", "Diagnosis + rule catalog", "Prioritizing what to inspect or test next"],
                    ["Race trial", "Short benchmark window", "Screening candidates before full measurement"],
                    ["Full tune trial", "Full benchmark window + guardrails", "Choosing the recommendation-only winner"],
                  ].map(([surface, source, use]) => (
                    <tr key={surface}>
                      <Td mono>{surface}</Td>
                      <Td>{source}</Td>
                      <Td>{use}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Diagnosis-focused candidates</H3>
              <P>
                The runner now focuses candidates from the baseline diagnosis when
                it can read one from <Code>derived/summary.json</Code>. Use{" "}
                <Code>--bottleneck</Code> to override that focus manually.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Bottleneck</Th>
                    <Th>Candidate family</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["input_bound", "DataLoader workers, pin_memory, prefetch_factor"],
                    ["copy_bound", "Pinned-memory-focused DataLoader candidates"],
                    ["launch_bound", "torch.compile and CUDA Graphs when --no-safe is enabled"],
                    ["memory_pressure", "CUDA allocator settings, then mixed precision when --no-safe is enabled"],
                    ["underutilized_gpu", "Batch size, mixed precision, then runtime candidates when --no-safe is enabled"],
                  ].map(([label, family]) => (
                    <tr key={label}>
                      <Td mono>{label}</Td>
                      <Td>{family}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Pre-run safety validation</H3>
              <P>
                Unsafe candidates are skipped before execution. They still get a
                trial directory with <Code>config.yaml</Code>,{" "}
                <Code>metrics.json</Code>, and <Code>stderr.log</Code> explaining
                the rejection reason.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Check</Th>
                    <Th>Rejects when</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Risk policy", "Candidate is high risk and --allow-risky-actions is not set"],
                    ["Batch size", "Memory headroom is below the safe threshold"],
                    ["Precision", "CUDA is unavailable, bf16 is unsupported, or quality checks are required"],
                    ["CUDA Graphs", "Shapes appear dynamic or CUDA is unavailable"],
                    ["torch.compile", "Compile is marked unsupported or dynamic behavior is incompatible"],
                  ].map(([check, rejects]) => (
                    <tr key={check}>
                      <Td mono>{check}</Td>
                      <Td>{rejects}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Benchmark window</H3>
              <P>
                Each trial writes an explicit <Code>benchmark_window.json</Code>.
                Metrics prefer <Code>measurement_window</Code> when per-step data
                is available, then fall back to <Code>steady_state</Code> and full
                run metrics.
              </P>
              <Pre>{`benchmark_window.json
{
  "warmup_steps": 5,
  "measurement_steps": 20,
  "repeat_count": 1,
  "timeout_s": 60
}`}</Pre>

              <H3>Env vars injected per trial</H3>
              <P>
                Each trial subprocess receives the standard{" "}
                <Code>FRX_*</Code> collect vars plus these tune-specific ones.
                The workload reads them to configure itself — see the SDK
                integration section for how to wire them up.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Variable</Th>
                    <Th>Set by</Th>
                    <Th>Purpose</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["FRX_TUNE_WARMUP_STEPS", "tune runner", "Steps to skip before measurement; workload should exit early"],
                    ["FRX_TUNE_MEASURE_STEPS", "tune runner", "Measurement steps requested"],
                    ["FRX_TUNE_MAX_STEPS", "tune runner", "Total steps (warmup + measure); workload exits at this count"],
                    ["FRX_TUNE_REPEAT_COUNT", "tune runner", "Repeat count used for noise-aware comparison"],
                    ["FRX_NUM_WORKERS", "dataloader tuner", "DataLoader num_workers value to use"],
                    ["FRX_PIN_MEMORY", "dataloader tuner", "'true' or 'false'"],
                    ["FRX_PREFETCH_FACTOR", "dataloader tuner", "DataLoader prefetch_factor value"],
                    ["FRX_PERSISTENT_WORKERS", "dataloader tuner", "'true' or 'false'"],
                    ["FRX_BATCH_SIZE", "batch size tuner", "Absolute batch size to use (Tier 1)"],
                    ["FRX_AMP_DTYPE", "mixed precision tuner", "'bfloat16' or 'float16' (Tier 1)"],
                    ["FRX_TORCH_COMPILE", "runtime tuner", "Enable torch.compile when supported"],
                    ["FRX_TORCH_COMPILE_MODE", "runtime tuner", "Compile mode such as reduce-overhead"],
                    ["FRX_CUDA_GRAPHS", "runtime tuner", "try_if_static_shapes"],
                    ["PYTORCH_CUDA_ALLOC_CONF", "memory tuner", "CUDA allocator configuration"],
                  ].map(([k, src, desc]) => (
                    <tr key={k}>
                      <Td mono>{k}</Td>
                      <Td>{src}</Td>
                      <Td>{desc}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Trial artifacts</H3>
              <Pre>{`runs/
  tune-<id>/
    baseline/
      config.yaml
      benchmark_window.json
      metrics.json
      stdout.log
      stderr.log
      derived/summary.json
      raw/trace.jsonl
    race/
      <candidate-id>/
        config.yaml
        benchmark_window.json
        metrics.json
        stdout.log
        stderr.log
    <candidate-id>/
      config.yaml
      benchmark_window.json
      metrics.json
      stdout.log
      stderr.log
    autopilot_report.json
    report.md`}</Pre>

              <H3>Workload integration</H3>
              <P>
                The workload reads the injected env vars and applies them. The
                minimal pattern for dataloader tuning:
              </P>
              <Pre>{`import os

num_workers     = int(os.environ.get("FRX_NUM_WORKERS", "4"))
pin_memory      = os.environ.get("FRX_PIN_MEMORY", "true") == "true"
prefetch_factor = int(os.environ.get("FRX_PREFETCH_FACTOR", "2"))
persistent      = os.environ.get("FRX_PERSISTENT_WORKERS", "true") == "true"
max_steps       = int(os.environ.get("FRX_TUNE_MAX_STEPS", "0")) or None

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent and num_workers > 0,
)

for step, batch in enumerate(loader):
    if max_steps and step >= max_steps:
        break
    # ... training step ...`}</Pre>

              <P>
                For AMP and batch size (<Code>--no-safe</Code>):
              </P>
              <Pre>{`import torch, os

amp_dtype_str = os.environ.get("FRX_AMP_DTYPE")          # "bfloat16" | "float16" | None
amp_dtype     = getattr(torch, amp_dtype_str, None) if amp_dtype_str else None
batch_size    = int(os.environ.get("FRX_BATCH_SIZE", "32"))

with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
    loss = model(batch)`}</Pre>

              <H3>Quality regression gates</H3>
              <P>
                A faster candidate is rejected if quality metrics regress. Loss is
                read from <Code>step_end.payload.loss</Code> when the workload
                emits it, and output drift checks are used when present in the
                summary quality fields.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Gate</Th>
                    <Th>Default</Th>
                    <Th>Flag</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Final loss vs baseline", "Reject if worse by more than 5%", "--max-final-loss-regression"],
                    ["Trial loss divergence", "Reject if final loss grows more than 50%", "--max-loss-divergence"],
                    ["NaN/Inf loss", "Reject", "--allow-nonfinite-loss"],
                    ["Output absolute drift", "Reject above 0.005 when reported", "--output-abs-tolerance"],
                  ].map(([gate, def_, flag]) => (
                    <tr key={gate}>
                      <Td>{gate}</Td>
                      <Td>{def_}</Td>
                      <Td mono>{flag}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <H3>Example output</H3>
              <Pre>{`frx autopilot — starting tune run tune-3f8a12b4
Workload : python train.py
Max trials: 12  |  Time budget: 60s/trial

Running baseline...
  Baseline: 4.8 steps/sec  (exit=0, steps=25)

Generated 8 candidates

Running quick race stage (1 warmup + 5 measure steps)...
  [1/8] race: dl:nw=0,pin=T ...
  [RACE]  dl:nw=0,pin=T                        +1.2%  (exit=0, steps=6)
  [2/8] race: dl:nw=2,pin=T ...
  [RACE]  dl:nw=2,pin=T                        +11.4% (exit=0, steps=6)
  [3/8] race: dl:nw=4,pin=T ...
  [RACE]  dl:nw=4,pin=T                        +19.3% (exit=0, steps=6)
  ...
Quick race promoted 3 of 8 candidates

  [1/3] full: dl:nw=4,pin=T ...
  [PASS]  dl:nw=4,pin=T                        +19.3% (exit=0, steps=25)
  [2/3] full: dl:nw=8,pin=T ...
  [PASS]  dl:nw=8,pin=T                        +18.1% (exit=0, steps=25)
  [3/3] full: amp:fp16 ...
  [FAIL]  amp:fp16                              +28.0% (exit=0, steps=25)
       ! quality regression: final loss 1.2 exceeds baseline 1 by more than 5%

Report saved: runs/tune-3f8a12b4/autopilot_report.json
Markdown report saved: runs/tune-3f8a12b4/report.md

──────────────────────────────────────────────────────────
  frx autopilot — Tune Report
  Job    : frx-tune
  Trials : 8 candidates + baseline
──────────────────────────────────────────────────────────

BASELINE
  Throughput   : 4.80 steps/sec
  Avg step     : 208.3 ms
  GPU util     : 1.3%
  Dominant stall: input_bound

TRIAL RESULTS
  dl:nw=4,pin=T                        +19.3% ✓
  dl:nw=8,pin=T                        +18.1% ✓
  dl:nw=4,pin=T                        [RACE] +19.3%  promoted to full benchmark
  dl:nw=8,pin=T                        [RACE] +18.1%  promoted to full benchmark
  dl:nw=2,pin=T                        [RACE] +11.4%  screened out by quicker candidates
  dl:nw=0,pin=T                        [RACE] +1.2%   screened out by quicker candidates

WINNER
  Config       : dl:nw=4,pin=T
  Throughput   : 5.73 steps/sec  (+19.3% vs baseline)
  Avg step     : 174.5 ms
  GPU util     : 4.1%

ENV VARS TO APPLY
  FRX_NUM_WORKERS=4
  FRX_PIN_MEMORY=true
  FRX_PERSISTENT_WORKERS=true
  FRX_PREFETCH_FACTOR=2

Applied: No — recommendation only
To apply: set the env vars above before launching your workload.`}</Pre>

              <H3>Promotion thresholds</H3>
              <P>
                A candidate is promoted only if it clears all of these. Noisy
                sub-threshold improvements are not recommended.
              </P>
              <Table>
                <thead>
                  <tr>
                    <Th>Guard</Th>
                    <Th>Default</Th>
                    <Th>Flag</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Minimum throughput improvement", "≥ 8%", "--min-speedup"],
                    ["Peak GPU memory ratio", "< 90%", "—"],
                    ["Step time regression", "< 10% worse than baseline", "—"],
                    ["Exit code", "0 (clean exit)", "—"],
                    ["Minimum steps captured", "≥ 3", "—"],
                    ["Quality gates", "Loss and numerics must pass", "quality flags above"],
                  ].map(([guard, def_, flag]) => (
                    <tr key={guard}>
                      <Td>{guard}</Td>
                      <Td mono>{def_}</Td>
                      <Td mono>{flag}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <div className="rounded-xl border border-violet-500/20 bg-violet-500/[0.05] p-4">
                <p className="text-sm font-semibold text-violet-300 mb-1">Current boundary</p>
                <p className="text-xs leading-5 text-slate-400">
                  Repeated trials now use median throughput and a measured noise
                  band. Interleaved ordering such as baseline A, trial, baseline B
                  is still future comparator work.
                </p>
              </div>
            </Section>

            {/* Bundle layout */}
            <Section id="bundle-layout">
              <div className="flex items-center gap-3">
                <H2>Bundle layout</H2>
                <FolderOpen className="h-5 w-5 text-slate-500" />
              </div>

              <P>
                Each <Code>collect</Code> run produces one directory under{" "}
                <Code>--out</Code> (default <Code>runs/</Code>) and a zip of it.
              </P>

              <Pre>{`runs/
  run-<id>/
    metadata.json            # Run metadata, artifact list, warnings
    manifest.json            # Included files, limited-data flag
    run_config.yaml          # Collector config + detected environment
    gpu_metrics.csv          # nvidia-smi samples (util %, memory, clocks)
    optional_logs.txt        # Combined workload stdout + stderr
    raw/
      trace.jsonl            # SDK event stream (one JSON object per line)
    derived/
      summary.json           # Pre-analyzed output — preferred by analyzer
    profiler/
      profiler_trace.json    # Chrome-format torch.profiler trace (imported)
  run-<id>.zip               # All of the above, zipped for upload`}</Pre>

              <H3>File roles</H3>
              <Table>
                <thead>
                  <tr>
                    <Th>File</Th>
                    <Th>Source</Th>
                    <Th>Required for analysis</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["derived/summary.json", "Generated by collect", "Preferred — fastest path"],
                    ["raw/trace.jsonl", "SDK auto-persist", "Yes, if no derived summary"],
                    ["profiler/profiler_trace.json", "Imported from --artifact-dir", "Fallback if no SDK trace"],
                    ["gpu_metrics.csv", "nvidia-smi poller", "Enriches GPU util data"],
                    ["metadata.json", "Generated by collect", "No (informational)"],
                    ["run_config.yaml", "Generated by collect", "No (informational)"],
                    ["optional_logs.txt", "Workload stdout/stderr", "No (debugging)"],
                  ].map(([f, s, r]) => (
                    <tr key={f}>
                      <Td mono>{f}</Td>
                      <Td>{s}</Td>
                      <Td>{r}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <P>
                The web analyzer scores bundle files when you upload multiple files
                at once. <Code>derived/summary.json</Code> scores highest (120 pts)
                and is used automatically when present.
              </P>
            </Section>

            {/* Analysis pipeline */}
            <Section id="analysis">
              <div className="flex items-center gap-3">
                <H2>Analysis pipeline</H2>
                <BarChart3 className="h-5 w-5 text-slate-500" />
              </div>

              <P>
                The analysis pipeline is pure Python with no GPU required. It
                accepts the SDK event stream or events reconstructed from a
                Chrome-format profiler trace, and produces a structured{" "}
                <Code>summary</Code> dict.
              </P>

              <Pre>{`from fournex.analysis import summarize_run_with_steady_state

summary = summarize_run_with_steady_state(events)
# summary["steady_state"]["diagnosis"]["user_facing_bottleneck"]
# → "input_bound"`}</Pre>

              <H3>Summary shape</H3>
              <Table>
                <thead>
                  <tr>
                    <Th>Key</Th>
                    <Th>Description</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["event_count", "Total events in the input stream"],
                    ["step_count", "Steps detected across the full run"],
                    ["selector", "steady_state window policy (skip_first_n, last_k)"],
                    ["run", "Scope object for all steps"],
                    ["steady_state", "Scope object for warm-up-excluded steps"],
                    ["scope_comparison", "Whether primary bottleneck changed between scopes"],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <Td mono>{k}</Td>
                      <Td>{v}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <P>
                Each scope object contains <Code>per_step</Code> (timing breakdown
                per step), <Code>run_summary</Code> (aggregated metrics),{" "}
                <Code>bottlenecks</Code> (scored list), and <Code>diagnosis</Code>{" "}
                (primary bottleneck + recommendations).
              </P>

              <H3>Symptom vs. root cause</H3>
              <P>
                <Code>underutilized_gpu</Code> often scores highest (the GPU is
                idle) but it is a symptom, not a cause. When a stall-type bottleneck
                (e.g. <Code>input_bound</Code>) is also present, the{" "}
                <Code>diagnosis.user_facing_bottleneck</Code> field is set to that
                root cause. The internal top signal is preserved in{" "}
                <Code>diagnosis.primary_bottleneck</Code>.
              </P>
              <Pre>{`{
  "primary_bottleneck":     "underutilized_gpu",   // internal top signal
  "user_facing_bottleneck": "input_bound",          // shown to users
  ...
}`}</Pre>
            </Section>

            {/* Bottleneck labels */}
            <Section id="bottlenecks">
              <div className="flex items-center gap-3">
                <H2>Bottleneck labels</H2>
                <Cpu className="h-5 w-5 text-slate-500" />
              </div>

              <Table>
                <thead>
                  <tr>
                    <Th>Label</Th>
                    <Th>Display name</Th>
                    <Th>Signal</Th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["input_bound", "Input Pipeline Starvation", "DataLoader wait ≥ 20% of step time"],
                    ["copy_bound", "Host-to-Device Copy Overhead", "H2D copy time ≥ 15% of step time"],
                    ["sync_bound", "Synchronization Overhead", "Sync wait ≥ 10% of step time"],
                    ["underutilized_gpu", "GPU Under-utilization", "GPU utilization < 35% (symptom)"],
                    ["memory_pressure", "Memory Pressure", "Peak memory ratio ≥ 90%"],
                    ["shape_instability", "Shape Instability", "Shape volatility ratio ≥ 30%"],
                    ["launch_bound", "Kernel Launch Overhead", "Profiler windows with many short kernels, stable shapes, low sampled util, and no dominant input/copy/sync stall"],
                    ["insufficient_telemetry", "Insufficient Telemetry", "No timing data and no GPU util samples"],
                  ].map(([label, display, signal]) => (
                    <tr key={label}>
                      <Td mono>{label}</Td>
                      <Td>{display}</Td>
                      <Td>{signal}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>

              <P>
                Labels are stable identifiers used in <Code>summary.json</Code>,
                CLI output, and the web frontend. The recommendation engine maps
                each label to a set of ranked fix cards.
              </P>
            </Section>

            {/* SDK integration */}
            <Section id="sdk">
              <div className="flex items-center gap-3">
                <H2>SDK integration</H2>
                <Package className="h-5 w-5 text-slate-500" />
              </div>

              <P>
                When <Code>collect</Code> wraps your training script, it sets{" "}
                <Code>FRX_AUTO_PERSIST=1</Code> and injects the output
                path. The SDK hooks emit events automatically if you use the
                provided context managers or callbacks.
              </P>

              <H3>PyTorch training loop</H3>
              <Pre>{`from fournex import AutopilotSession

session = AutopilotSession.from_env()   # reads FRX_* env vars

for epoch in range(num_epochs):
    for batch in dataloader:
        with session.step(step_id=global_step, step_kind="train"):
            with session.dataloader_span():
                batch = next_batch()    # already inside dataloader loop
            with session.forward_span():
                loss = model(batch)
            with session.backward_span():
                loss.backward()
            with session.optimizer_span():
                optimizer.step()
        global_step += 1

session.flush()`}</Pre>

              <P>
                If you already use <Code>torch.profiler</Code>, you can skip the
                SDK entirely and point <Code>--artifact-dir</Code> at the directory
                where the profiler writes its Chrome-format trace. The CLI will
                import and analyze it automatically.
              </P>

              <H3>Profiler-only workflow</H3>
              <Pre>{`# In your training script, write the profiler trace to frx-job-run/
profiler = torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("frx-job-run"),
)

# Then collect — the CLI imports profiler_trace.json automatically
frx collect -- python train.py

# If the trace is written somewhere else, pass that directory explicitly
frx collect --artifact-dir gpu-job-run-tiny-kernels -- python train.py`}</Pre>

              <div className="mt-6 rounded-xl border border-amber-500/20 bg-amber-500/[0.06] p-4">
                <p className="text-sm font-semibold text-amber-300 mb-1">Note on data richness</p>
                <p className="text-xs leading-5 text-slate-400">
                  SDK instrumentation produces the richest data: exact step
                  boundaries, DataLoader wait times, and H2D copy spans are
                  recorded precisely. Profiler-only mode reconstructs these from
                  Chrome trace heuristics and may have lower confidence on some
                  bottleneck types.
                </p>
              </div>
            </Section>

            {/* Footer links */}
            <div className="border-t border-white/10 pt-8 flex flex-wrap gap-4">
              <Link
                href="/analyze"
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-5 py-3 text-sm font-semibold text-white transition hover:from-violet-500 hover:to-fuchsia-400"
              >
                Open analyzer
                <ChevronRight className="h-4 w-4" />
              </Link>
              <Link
                href="/how-it-works"
                className="inline-flex items-center gap-2 rounded-xl border border-white/12 bg-white/[0.03] px-5 py-3 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/[0.06]"
              >
                How it works
              </Link>
            </div>

          </div>
        </div>
      </div>
    </main>
  );
}
