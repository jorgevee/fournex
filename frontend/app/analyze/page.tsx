"use client";

import { useState, useCallback } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardPaste,
  Loader2,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────

interface Recommendation {
  id: string;
  title: string;
  priority: "high" | "medium" | "low";
  score: number;
  confidence: number;
  expected_impact: string;
  effort: string;
  category: string;
  why: string;
  actions: string[];
  validation: string[];
  risks: string[];
  triggered_by: string;
}

interface Bundle {
  label: string;
  category: string;
  recommendation_ids: string[];
}

interface Diagnosis {
  primary_bottleneck: string | null;
  secondary_bottlenecks: string[];
  confidence: { level: string; score: number; reason: string };
  evidence: Record<string, unknown>;
  why: string[];
  why_not_others: string[];
  recommendations: Recommendation[];
  recommendation_bundles: Bundle[];
  dominant_stall_type: string;
  classifier_version: string;
}

interface Summary {
  run_id?: string;
  event_count: number;
  step_count: number;
  run_summary: Record<string, unknown>;
  bottlenecks: Array<{ label: string; score: number; evidence: Record<string, unknown> }>;
  diagnosis: Diagnosis;
  scope: { name: string; step_ids: number[] };
}

// ── Constants ──────────────────────────────────────────────────────────────

const DEMO_SUMMARY: Summary = {
  run_id: "demo-abc123",
  event_count: 14,
  step_count: 2,
  scope: { name: "run", step_ids: [1, 2] },
  run_summary: {
    average_gpu_utilization_pct: 41,
    dominant_stall_type: "input_bound",
    step_time_avg_ns: 100,
    throughput_steps_per_sec: 10000000,
  },
  bottlenecks: [
    {
      label: "input_bound",
      score: 0.325,
      evidence: { avg_dataloader_fraction: 0.325, dominant_stall_type: "input_bound" },
    },
  ],
  diagnosis: {
    primary_bottleneck: "input_bound",
    secondary_bottlenecks: [],
    confidence: {
      level: "medium",
      score: 0.725,
      reason: "input_bound is the only bottleneck above threshold.",
    },
    evidence: { avg_dataloader_fraction: 0.325 },
    why: [
      "Average DataLoader wait fraction is 0.325.",
      "Run summary dominant stall type is input_bound.",
    ],
    why_not_others: [],
    dominant_stall_type: "input_bound",
    classifier_version: "0.2.0",
    recommendation_bundles: [
      {
        label: "Input Pipeline Optimization",
        category: "input_pipeline",
        recommendation_ids: ["rec_input_num_workers", "rec_input_pinned_memory", "rec_input_prefetch_factor"],
      },
    ],
    recommendations: [
      {
        id: "rec_input_num_workers",
        title: "Increase DataLoader workers",
        priority: "medium",
        score: 0.58,
        confidence: 0.325,
        expected_impact: "high",
        effort: "low",
        category: "input_pipeline",
        why: "DataLoader wait accounts for a significant fraction of step time, suggesting the input pipeline cannot keep pace with the GPU.",
        actions: [
          "Benchmark num_workers at 4, 8, and 16 (relative to available CPU cores).",
          "Start with num_workers = cpu_count // 2 and tune from there.",
          "Watch for step-time variance and CPU memory pressure as workers increase.",
        ],
        validation: [
          "GPU active fraction should rise after increasing workers.",
          "DataLoader wait fraction should drop below 0.10.",
          "Step-time standard deviation should decrease.",
        ],
        risks: [
          "Too many workers can cause CPU contention or excessive RAM pressure.",
          "On machines with few cores, gains plateau quickly.",
        ],
        triggered_by: "rule_input_starvation_moderate",
      },
      {
        id: "rec_input_pinned_memory",
        title: "Enable pinned memory for DataLoader",
        priority: "medium",
        score: 0.525,
        confidence: 0.325,
        expected_impact: "medium",
        effort: "low",
        category: "input_pipeline",
        why: "DataLoader wait accounts for a significant fraction of step time, suggesting the input pipeline cannot keep pace with the GPU.",
        actions: [
          "Set pin_memory=True in DataLoader constructor.",
          "Verify the host has enough page-locked memory available.",
        ],
        validation: [
          "H2D copy time should decrease after enabling pin_memory.",
          "GPU active fraction should improve slightly.",
        ],
        risks: ["Pinned memory is limited; too many workers with pin_memory can exhaust it."],
        triggered_by: "rule_input_starvation_moderate",
      },
      {
        id: "rec_input_prefetch_factor",
        title: "Increase DataLoader prefetch factor",
        priority: "medium",
        score: 0.525,
        confidence: 0.325,
        expected_impact: "medium",
        effort: "low",
        category: "input_pipeline",
        why: "DataLoader wait accounts for a significant fraction of step time.",
        actions: [
          "Set prefetch_factor=2 or 4 (requires num_workers > 0).",
          "Monitor RAM usage — prefetching increases memory held in worker queues.",
        ],
        validation: [
          "Batch-ready gaps between steps should shrink.",
          "DataLoader wait fraction should decrease.",
        ],
        risks: ["prefetch_factor requires num_workers >= 1."],
        triggered_by: "rule_input_starvation_moderate",
      },
    ],
  },
};

// ── Helpers ────────────────────────────────────────────────────────────────

function priorityStyle(p: string) {
  if (p === "high") return "border-red-400/30 bg-red-400/10 text-red-300";
  if (p === "medium") return "border-yellow-400/30 bg-yellow-400/10 text-yellow-300";
  return "border-slate-400/30 bg-slate-400/10 text-slate-400";
}

function confidenceStyle(level: string) {
  if (level === "high") return "text-emerald-400";
  if (level === "medium") return "text-yellow-400";
  return "text-slate-400";
}

function bottleneckLabel(key: string | null) {
  const map: Record<string, string> = {
    input_bound: "Input Pipeline Starvation",
    copy_bound: "Host-to-Device Copy Overhead",
    sync_bound: "Synchronization Overhead",
    underutilized_gpu: "GPU Under-utilization",
    memory_pressure: "Memory Pressure",
    shape_instability: "Shape Instability",
    launch_bound: "Kernel Launch Overhead",
    insufficient_telemetry: "Insufficient Telemetry",
  };
  return key ? (map[key] ?? key.replace(/_/g, " ")) : "No bottleneck detected";
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RecCard({
  rec,
  runId,
  feedback,
  onFeedback,
}: {
  rec: Recommendation;
  runId: string;
  feedback: Record<string, string>;
  onFeedback: (recId: string, outcome: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const given = feedback[rec.id];

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden">
      {/* Header row */}
      <div className="flex items-start justify-between gap-4 px-5 py-4">
        <div className="flex items-start gap-3 min-w-0">
          <span
            className={`mt-0.5 shrink-0 rounded-full border px-2 py-0.5 text-[0.6rem] font-semibold uppercase tracking-[0.18em] ${priorityStyle(rec.priority)}`}
          >
            {rec.priority}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-white">{rec.title}</p>
            <p className="mt-0.5 text-xs leading-5 text-slate-400 line-clamp-2">{rec.why}</p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {/* Feedback */}
          <button
            onClick={() => onFeedback(rec.id, "accepted")}
            title="Helpful"
            className={`rounded-lg p-1.5 transition ${
              given === "accepted"
                ? "bg-emerald-400/20 text-emerald-400"
                : "text-slate-500 hover:text-emerald-400 hover:bg-emerald-400/10"
            }`}
          >
            <ThumbsUp size={14} />
          </button>
          <button
            onClick={() => onFeedback(rec.id, "rejected")}
            title="Not helpful"
            className={`rounded-lg p-1.5 transition ${
              given === "rejected"
                ? "bg-red-400/20 text-red-400"
                : "text-slate-500 hover:text-red-400 hover:bg-red-400/10"
            }`}
          >
            <ThumbsDown size={14} />
          </button>
          {/* Expand toggle */}
          <button
            onClick={() => setExpanded((v) => !v)}
            className="rounded-lg p-1.5 text-slate-500 hover:text-white transition"
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-white/10 px-5 py-4 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Section title="Actions">
              <ol className="list-decimal list-inside space-y-1">
                {rec.actions.map((a, i) => (
                  <li key={i} className="text-xs text-slate-300">{a}</li>
                ))}
              </ol>
            </Section>
            <Section title="How to verify">
              <ul className="space-y-1">
                {rec.validation.map((v, i) => (
                  <li key={i} className="flex gap-2 text-xs text-slate-300">
                    <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-emerald-500" />
                    {v}
                  </li>
                ))}
              </ul>
            </Section>
          </div>
          {rec.risks.length > 0 && (
            <Section title="Risks">
              <ul className="space-y-1">
                {rec.risks.map((r, i) => (
                  <li key={i} className="flex gap-2 text-xs text-slate-400">
                    <AlertCircle size={12} className="mt-0.5 shrink-0 text-yellow-500" />
                    {r}
                  </li>
                ))}
              </ul>
            </Section>
          )}
          <div className="flex gap-4 text-[0.65rem] text-slate-500">
            <span>Impact: <span className="text-slate-300 capitalize">{rec.expected_impact}</span></span>
            <span>Effort: <span className="text-slate-300 capitalize">{rec.effort}</span></span>
            <span>Score: <span className="text-slate-300">{rec.score.toFixed(2)}</span></span>
          </div>
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-2 text-[0.6rem] font-semibold uppercase tracking-[0.2em] text-slate-500">{title}</p>
      {children}
    </div>
  );
}

// ── Report ─────────────────────────────────────────────────────────────────

function Report({ summary }: { summary: Summary }) {
  const { diagnosis } = summary;
  const runId = summary.run_id ?? `run-${Date.now()}`;
  const [feedback, setFeedback] = useState<Record<string, string>>({});

  const handleFeedback = useCallback(async (recId: string, outcome: string) => {
    setFeedback((prev) => ({ ...prev, [recId]: outcome }));
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, rec_id: recId, outcome }),
      });
    } catch {
      // feedback is best-effort; don't block the UI
    }
  }, [runId]);

  const recById = Object.fromEntries(diagnosis.recommendations.map((r) => [r.id, r]));

  // Collect rec IDs already in bundles to avoid duplication
  const bundledIds = new Set(diagnosis.recommendation_bundles.flatMap((b) => b.recommendation_ids));
  const unbundled = diagnosis.recommendations.filter((r) => !bundledIds.has(r.id));

  return (
    <div className="space-y-8">
      {/* Bottleneck header */}
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.22em] text-slate-500">
              Primary bottleneck
            </p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-white">
              {bottleneckLabel(diagnosis.primary_bottleneck)}
            </h2>
            {diagnosis.secondary_bottlenecks.length > 0 && (
              <p className="mt-1 text-xs text-slate-400">
                Also detected: {diagnosis.secondary_bottlenecks.map(bottleneckLabel).join(", ")}
              </p>
            )}
          </div>
          <div className="text-right">
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.22em] text-slate-500">
              Confidence
            </p>
            <p className={`mt-1 text-xl font-semibold capitalize ${confidenceStyle(diagnosis.confidence.level)}`}>
              {diagnosis.confidence.level}
            </p>
            <p className="text-xs text-slate-500">{(diagnosis.confidence.score * 100).toFixed(0)}%</p>
          </div>
        </div>

        {/* Why bullets */}
        {diagnosis.why.length > 0 && (
          <ul className="mt-5 space-y-1.5 border-t border-white/10 pt-4">
            {diagnosis.why.map((bullet, i) => (
              <li key={i} className="flex gap-2 text-sm text-slate-300">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
                {bullet}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Bundles */}
      {diagnosis.recommendation_bundles.map((bundle) => {
        const recs = bundle.recommendation_ids.map((id) => recById[id]).filter(Boolean);
        if (!recs.length) return null;
        return (
          <div key={bundle.category}>
            <div className="mb-3 flex items-center gap-3">
              <p className="text-sm font-semibold text-white">{bundle.label}</p>
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[0.6rem] text-slate-400">
                {recs.length} fix{recs.length !== 1 ? "es" : ""}
              </span>
            </div>
            <div className="space-y-2">
              {recs.map((rec) => (
                <RecCard
                  key={rec.id}
                  rec={rec}
                  runId={runId}
                  feedback={feedback}
                  onFeedback={handleFeedback}
                />
              ))}
            </div>
          </div>
        );
      })}

      {/* Unbundled recommendations */}
      {unbundled.length > 0 && (
        <div>
          <p className="mb-3 text-sm font-semibold text-white">Additional recommendations</p>
          <div className="space-y-2">
            {unbundled.map((rec) => (
              <RecCard
                key={rec.id}
                rec={rec}
                runId={runId}
                feedback={feedback}
                onFeedback={handleFeedback}
              />
            ))}
          </div>
        </div>
      )}

      {diagnosis.recommendations.length === 0 && (
        <p className="text-sm text-slate-400">No recommendations generated for this profile.</p>
      )}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function AnalyzePage() {
  const [raw, setRaw] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function parseAndDisplay(json: string) {
    try {
      const parsed = JSON.parse(json) as Summary;
      if (!parsed.diagnosis) throw new Error("Missing 'diagnosis' field — paste the output of summarize_run().");
      setSummary(parsed);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid JSON");
      setSummary(null);
    }
  }

  async function handleAnalyze() {
    if (!raw.trim()) return;
    setLoading(true);
    setError(null);

    // Try to detect whether this is a raw-events payload or a pre-analyzed summary
    try {
      const parsed = JSON.parse(raw);

      // If it has a diagnosis key already, render directly without calling the API
      if (parsed.diagnosis) {
        setSummary(parsed as Summary);
        setLoading(false);
        return;
      }

      // Otherwise treat as { events, environment } and call the API
      const body = Array.isArray(parsed) ? { events: parsed } : parsed;
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "API error");
      }
      const data = await res.json() as Summary;
      setSummary(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to analyze");
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }

  function loadDemo() {
    const json = JSON.stringify(DEMO_SUMMARY, null, 2);
    setRaw(json);
    parseAndDisplay(json);
  }

  return (
    <main className="min-h-screen bg-[#020617] text-white">
      <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-400/25 bg-blue-400/10 px-3 py-1 text-sm font-medium text-blue-300">
            <Zap size={13} />
            Autopilot Analyzer
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Profile your GPU run
          </h1>
          <p className="mt-3 text-base text-slate-400">
            Paste a run summary JSON (output of <code className="rounded bg-white/5 px-1 py-0.5 font-mono text-xs text-slate-200">summarize_run()</code>) or a raw events array. Fournex will name the bottleneck and surface the highest-ROI fixes.
          </p>
        </div>

        {/* Input area */}
        {!summary && (
          <div className="space-y-4">
            <textarea
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
              placeholder='{ "events": [...] }  or paste a summarize_run() output'
              rows={12}
              className="w-full rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 font-mono text-xs text-slate-300 placeholder-slate-600 focus:border-blue-500/50 focus:outline-none focus:ring-1 focus:ring-blue-500/30 resize-y"
            />
            {error && (
              <p className="flex items-center gap-2 text-sm text-red-400">
                <AlertCircle size={14} /> {error}
              </p>
            )}
            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleAnalyze}
                disabled={!raw.trim() || loading}
                className="inline-flex items-center gap-2 rounded-full bg-blue-500 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:opacity-40"
              >
                {loading ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
                Analyze
              </button>
              <button
                onClick={loadDemo}
                className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/10"
              >
                <ClipboardPaste size={15} />
                Load demo
              </button>
            </div>
          </div>
        )}

        {/* Report */}
        {summary && (
          <>
            <div className="mb-6 flex items-center justify-between">
              <div className="text-xs text-slate-500">
                {summary.step_count} step{summary.step_count !== 1 ? "s" : ""} · {summary.event_count} events · scope: {summary.scope.name}
              </div>
              <button
                onClick={() => { setSummary(null); setRaw(""); setError(null); }}
                className="text-xs text-slate-500 underline underline-offset-2 hover:text-slate-300 transition"
              >
                Analyze another
              </button>
            </div>
            <Report summary={summary} />
          </>
        )}
      </div>
    </main>
  );
}
