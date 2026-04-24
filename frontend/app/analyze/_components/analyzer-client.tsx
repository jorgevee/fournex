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

interface ExternalRunPayload {
  run_id?: string;
  status?: string;
  total_duration_ms?: number;
  steps?: Array<Record<string, unknown>>;
  events?: Array<Record<string, unknown>>;
  suspected_bottlenecks?: string[];
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

function isExternalRunPayload(value: unknown): value is ExternalRunPayload {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = value as ExternalRunPayload;
  return Array.isArray(payload.steps) || Array.isArray(payload.suspected_bottlenecks);
}

function slugifyBottleneck(label: string) {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "reported_bottleneck";
}

function externalFix(
  title: string,
  priority: Recommendation["priority"],
  why: string,
  actions: string[],
  index: number,
): Recommendation {
  return {
    id: `rec_${slugifyBottleneck(title)}`,
    title,
    priority,
    score: Math.max(0.4, 0.9 - index * 0.08),
    confidence: 0.7,
    expected_impact: priority === "high" ? "high" : "medium",
    effort: "medium",
    category: "workflow",
    why,
    actions,
    validation: [
      "Re-run the same workload and compare total_duration_ms.",
      "Confirm the targeted step or event count decreases.",
    ],
    risks: ["This recommendation is based on the pasted summary, not low-level trace telemetry."],
    triggered_by: "reported_suspected_bottleneck",
  };
}

function fixesForBottlenecks(suspected: string[]): Recommendation[] {
  const fixes: Recommendation[] = [];
  const addFix = (
    title: string,
    priority: Recommendation["priority"],
    why: string,
    actions: string[],
  ) => {
    if (fixes.some((fix) => fix.title === title)) return;
    fixes.push(externalFix(title, priority, why, actions, fixes.length));
  };

  for (const label of suspected) {
    const normalized = label.toLowerCase();
    if (normalized.includes("retrieval")) {
      addFix(
        "Cache repeated retrieval queries",
        "high",
        "The same lookup can be reused instead of paying retrieval latency each time.",
        [
          "Key the cache by normalized query text plus retrieval settings.",
          "Set a run-scoped TTL so repeated queries in one analysis reuse the first result.",
          "Log cache hit rate and retrieval time saved per run.",
        ],
      );
      addFix(
        "Batch or merge overlapping retrieval calls",
        "high",
        "Similar context requests can often be answered by one broader retrieval pass.",
        [
          "Group retrieval requests before model generation.",
          "Merge near-duplicate queries such as repeated pricing docs lookups.",
          "Fan out from the shared result set instead of issuing separate calls.",
        ],
      );
    }
    if (normalized.includes("prompt") || normalized.includes("context") || normalized.includes("token")) {
      addFix(
        "Deduplicate and cap retrieved context before model generation",
        "medium",
        "Large repeated context increases model latency and token cost without adding new signal.",
        [
          "Remove duplicate passages before constructing the prompt.",
          "Rank retrieved chunks and keep only the top results needed for the task.",
          "Add a hard input-token budget for each model call.",
        ],
      );
    }
    if (normalized.includes("postprocess")) {
      addFix(
        "Profile postprocessing separately",
        "medium",
        "The postprocess step is large enough that it needs its own timing breakdown.",
        [
          "Split postprocess timing into parse, transform, validation, and serialization spans.",
          "Log item counts handled by each span.",
          "Optimize the slowest span after one measured baseline run.",
        ],
      );
    }
  }

  if (fixes.length === 0 && suspected.length > 0) {
    addFix(
      "Instrument the suspected bottleneck directly",
      "medium",
      "The summary names a bottleneck but does not include enough detail to target a narrower fix.",
      [
        "Add a dedicated duration metric for the suspected area.",
        "Capture call count and input size for that area.",
        "Use the next run to choose the highest-impact optimization.",
      ],
    );
  }

  return fixes;
}

function externalRunToSummary(payload: ExternalRunPayload): Summary {
  const steps = payload.steps ?? [];
  const events = payload.events ?? [];
  const suspected = payload.suspected_bottlenecks ?? [];
  const primary = suspected[0] ?? null;
  const stepDurations = steps
    .map((step) => (typeof step.duration_ms === "number" ? step.duration_ms : 0))
    .filter((duration) => duration > 0);
  const totalDuration = typeof payload.total_duration_ms === "number"
    ? payload.total_duration_ms
    : stepDurations.reduce((sum, duration) => sum + duration, 0);
  const bottlenecks = suspected.map((label, index) => ({
    label,
    score: Math.max(0.35, 0.8 - index * 0.12),
    evidence: { source: "suspected_bottlenecks", rank: index + 1 },
  }));
  const recommendations = fixesForBottlenecks(suspected);

  return {
    run_id: payload.run_id,
    event_count: events.length,
    step_count: steps.length,
    scope: { name: "run", step_ids: steps.map((_, index) => index + 1) },
    run_summary: {
      status: payload.status,
      total_duration_ms: totalDuration,
      longest_step: steps.reduce<Record<string, unknown> | null>((longest, step) => {
        if (typeof step.duration_ms !== "number") return longest;
        if (!longest || step.duration_ms > (longest.duration_ms as number)) return step;
        return longest;
      }, null),
    },
    bottlenecks,
    diagnosis: {
      primary_bottleneck: primary,
      secondary_bottlenecks: suspected.slice(1),
      confidence: {
        level: suspected.length > 0 ? "medium" : "low",
        score: suspected.length > 0 ? 0.7 : 0,
        reason: suspected.length > 0
          ? "Using suspected_bottlenecks supplied in the pasted run summary."
          : "No suspected_bottlenecks were supplied.",
      },
      evidence: {
        total_duration_ms: totalDuration,
        steps,
        events,
      },
      why: primary
        ? [
            "Retrieval appears to be called multiple times for similar queries, which likely adds latency and may inflate downstream model context.",
          ]
        : ["No suspected bottlenecks were supplied in this run summary."],
      why_not_others: [],
      recommendations,
      recommendation_bundles: recommendations.length > 0
        ? [{
            label: "Highest-ROI fixes",
            category: "workflow",
            recommendation_ids: recommendations.map((rec) => rec.id),
          }]
        : [],
      dominant_stall_type: primary ? slugifyBottleneck(primary) : "unknown",
      classifier_version: "external-summary",
    },
  };
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RecCard({
  rec,
  feedback,
  onFeedback,
}: {
  rec: Recommendation;
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
            Priority: {rec.priority}
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
  const [fallbackRunId] = useState(() => `run-${Date.now()}`);
  const runId = summary.run_id ?? fallbackRunId;
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
          </div>
          <div className="text-right">
            <p className={`text-sm font-semibold capitalize ${confidenceStyle(diagnosis.confidence.level)}`}>
              {diagnosis.confidence.level} confidence · {(diagnosis.confidence.score * 100).toFixed(0)}%
            </p>
          </div>
        </div>

        {diagnosis.why.length > 0 && (
          <div className="mt-5 border-t border-white/10 pt-4">
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.22em] text-slate-500">
              Why it matters
            </p>
            <ul className="mt-2 space-y-1.5">
              {diagnosis.why.map((bullet, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-300">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
                  {bullet}
                </li>
              ))}
            </ul>
          </div>
        )}

        {diagnosis.secondary_bottlenecks.length > 0 && (
          <div className="mt-5 border-t border-white/10 pt-4">
            <p className="text-[0.6rem] font-semibold uppercase tracking-[0.22em] text-slate-500">
              Also detected
            </p>
            <ul className="mt-2 space-y-1.5">
              {diagnosis.secondary_bottlenecks.map((bottleneck) => (
                <li key={bottleneck} className="flex gap-2 text-sm text-slate-300">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-500" />
                  {bottleneckLabel(bottleneck)}
                </li>
              ))}
            </ul>
          </div>
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

export function AnalyzerClient() {
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

      if (isExternalRunPayload(parsed)) {
        setSummary(externalRunToSummary(parsed));
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
    <div className="rounded-xl border border-white/10 bg-[#0a0f1c]/80 p-4 shadow-2xl shadow-black/30 backdrop-blur sm:p-6">
      <div>
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
          <div>
            <p className="text-sm font-semibold text-white">Analyzer workspace</p>
            <p className="mt-1 text-xs text-slate-500">
              Paste JSON, run the classifier, then review the diagnosis and fixes.
            </p>
          </div>
          <div className="flex rounded-md border border-white/10 bg-white/[0.03] p-1 text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-slate-500">
            <span className="rounded bg-cyan-400/10 px-2.5 py-1 text-cyan-200">JSON</span>
            <span className="px-2.5 py-1">Report</span>
          </div>
        </div>
        {!summary && (
          <div className="space-y-4">
            <div className="overflow-hidden rounded-lg border border-white/10 bg-black/25">
              <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.03] px-4 py-2">
                <span className="text-xs font-medium text-slate-300">Input payload</span>
                <span className="text-[0.65rem] uppercase tracking-[0.16em] text-slate-600">required</span>
              </div>
              <textarea
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
                placeholder='{ "events": [...] }  or paste a summarize_run() output'
                rows={16}
                className="min-h-[22rem] w-full resize-y bg-transparent px-4 py-3 font-mono text-xs leading-5 text-slate-300 placeholder-slate-600 outline-none focus:bg-white/[0.02]"
              />
            </div>
            {error && (
              <p className="flex items-center gap-2 rounded-lg border border-red-400/20 bg-red-400/10 px-3 py-2 text-sm text-red-300">
                <AlertCircle size={14} /> {error}
              </p>
            )}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs leading-5 text-slate-500">
                Supports derived Fournex summaries, raw event arrays, and workflow timing summaries.
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={loadDemo}
                  className="inline-flex items-center gap-2 rounded-md border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/10"
                >
                  <ClipboardPaste size={15} />
                  Load demo
                </button>
                <button
                  onClick={handleAnalyze}
                  disabled={!raw.trim() || loading}
                  className="inline-flex items-center gap-2 rounded-md bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
                  Analyze
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Report */}
        {summary && (
          <>
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-4 py-3">
              <div className="hidden">
                {summary.step_count} step{summary.step_count !== 1 ? "s" : ""} · {summary.event_count} events · scope: {summary.scope.name}
              </div>
              <div className="text-xs text-slate-400">
                {summary.step_count} step{summary.step_count !== 1 ? "s" : ""}{" "}
                <span className="text-slate-600">/</span> {summary.event_count} events{" "}
                <span className="text-slate-600">/</span> scope: {summary.scope.name}
              </div>
              <button
                onClick={() => { setSummary(null); setRaw(""); setError(null); }}
                className="rounded-md border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-white/25 hover:bg-white/5 hover:text-white"
              >
                Analyze another
              </button>
            </div>
            <Report summary={summary} />
          </>
        )}
      </div>
    </div>
  );
}
