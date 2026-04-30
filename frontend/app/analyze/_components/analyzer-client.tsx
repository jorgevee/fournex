"use client";

import { useState, useCallback } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardPaste,
  FolderUp,
  Gauge,
  Loader2,
  Target,
  ThumbsDown,
  ThumbsUp,
  TrendingUp,
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
  user_facing_bottleneck?: string | null;
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

interface ProfilerTraceEvent {
  ph?: string;
  cat?: string;
  name?: string;
  ts?: number;
  dur?: number;
  args?: Record<string, unknown>;
}

interface ProfilerTracePayload {
  trace_id?: string;
  traceEvents: ProfilerTraceEvent[];
}

interface BundleStatus {
  fileNames: string[];
  selectedFile: string | null;
  hasSummary: boolean;
  hasMetadata: boolean;
  hasProfilerTrace: boolean;
  hasGpuMetrics: boolean;
  hasRunConfig: boolean;
  hasLogs: boolean;
  limited: boolean;
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

function healthColor(score: number) {
  if (score >= 75) return "text-emerald-300";
  if (score >= 50) return "text-yellow-300";
  return "text-red-300";
}

function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function displayText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value.map(displayText).filter(Boolean).join(", ");
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["label", "title", "name", "message", "reason", "description"]) {
      if (typeof record[key] === "string") return record[key];
    }
    try {
      return JSON.stringify(value);
    } catch {
      return "Structured details available in raw data";
    }
  }
  return String(value);
}

function pct(value: number) {
  return `${Math.round(value)}%`;
}

function formatDuration(ms: number | null) {
  if (ms === null) return "Unknown";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms >= 10) return `${Math.round(ms)}ms`;
  return `${ms.toFixed(1)}ms`;
}

function durationMsFromSummary(summary: Summary) {
  const avgNs = numericValue(summary.run_summary.step_time_avg_ns);
  if (avgNs !== null) return avgNs / 1_000_000;
  const totalMs = numericValue(summary.run_summary.total_duration_ms);
  if (totalMs !== null && summary.step_count > 0) return totalMs / summary.step_count;
  return totalMs;
}

function isAnalysisIncomplete(summary: Summary) {
  if (isExternalWorkflowSummary(summary)) return false;
  const hasPrimary = Boolean(summary.diagnosis.primary_bottleneck || summary.diagnosis.user_facing_bottleneck);
  const avgNs = numericValue(summary.run_summary.step_time_avg_ns) ?? 0;
  return !hasPrimary && summary.bottlenecks.length === 0 && (summary.step_count === 0 || avgNs <= 0);
}

function incompleteAnalysisReason(summary: Summary) {
  if (summary.event_count <= 0) {
    return "The selected payload did not include trace events. Include derived/summary.json, raw/trace.jsonl, or profiler/profiler_trace.json from the collected run.";
  }
  if (summary.step_count <= 0) {
    return "Trace events were present, but no completed training steps were detected. Add ProfilerStep#N events or wrap each iteration with a top-level record_function.";
  }
  return "The run has zero measured step time, so bottleneck scores would be misleading. Re-collect with timing events or import the profiler trace into the bundle.";
}

function topBottleneckScore(summary: Summary) {
  return summary.bottlenecks[0]?.score ?? summary.diagnosis.confidence.score ?? 0;
}

function isExternalWorkflowSummary(summary: Summary) {
  return summary.diagnosis.classifier_version === "external-summary";
}

function speedupRange(summary: Summary) {
  const primary = slugifyBottleneck(summary.diagnosis.primary_bottleneck ?? "");
  const highPriorityCount = summary.diagnosis.recommendations.filter((rec) => rec.priority === "high").length;
  if (primary.includes("input_bound")) return [1.4, 2.0] as const;
  if (primary.includes("copy_bound")) return [1.15, 1.45] as const;
  if (primary.includes("sync_bound")) return [1.2, 1.6] as const;
  if (primary.includes("launch_bound")) return [1.25, 1.8] as const;
  if (primary.includes("memory_pressure")) return [1.1, 1.35] as const;
  if (primary.includes("search") || primary.includes("retrieval")) return [1.25, 1.7] as const;
  if (highPriorityCount >= 2) return [1.3, 1.8] as const;
  return [1.15, 1.45] as const;
}

function isLaunchBoundSummary(summary: Summary) {
  const labels = [
    summary.diagnosis.primary_bottleneck,
    summary.diagnosis.user_facing_bottleneck,
    ...summary.bottlenecks.map((item) => item.label),
  ].map((value) => slugifyBottleneck(displayText(value)));
  return labels.some((label) => label.includes("launch_bound") || label.includes("kernel_launch"));
}

function reportMetrics(summary: Summary) {
  const externalWorkflow = isExternalWorkflowSummary(summary);
  const launchBound = isLaunchBoundSummary(summary);
  const gpuActive = numericValue(summary.run_summary.average_gpu_utilization_pct);
  const bottleneckScore = topBottleneckScore(summary);
  const hasInputWait = summary.diagnosis.primary_bottleneck === "input_bound" ||
    summary.run_summary.dominant_stall_type === "input_bound";
  const gpuActiveUnreliable = (hasInputWait || launchBound) && gpuActive !== null && gpuActive <= 1;
  const healthScore = externalWorkflow
    ? Math.round(Math.max(0, 100 - bottleneckScore * 100))
    : launchBound
    ? Math.round(Math.max(0, 100 - bottleneckScore * 100))
    : gpuActive !== null && !gpuActiveUnreliable
    ? Math.round(gpuActive)
    : Math.round(Math.max(0, 100 - bottleneckScore * 100));
  const waitWaste = hasInputWait ? Math.round(bottleneckScore * 100) : 0;
  const waste = gpuActive !== null && !hasInputWait && !launchBound && !gpuActiveUnreliable
    ? Math.max(0, 100 - gpuActive)
    : Math.max(waitWaste, Math.round(bottleneckScore * 100));
  const [speedupLow, speedupHigh] = speedupRange(summary);
  const currentMs = durationMsFromSummary(summary);
  const projectedHighMs = currentMs === null ? null : currentMs / speedupLow;
  const projectedLowMs = currentMs === null ? null : currentMs / speedupHigh;
  const projectedUtilLow = gpuActive === null || gpuActiveUnreliable || launchBound ? null : Math.max(
    Math.round(gpuActive),
    Math.min(100, Math.round(gpuActive * speedupLow)),
  );
  const projectedUtilHigh = gpuActive === null || gpuActiveUnreliable || launchBound ? null : Math.max(
    projectedUtilLow ?? Math.round(gpuActive),
    Math.min(100, Math.round(gpuActive * speedupHigh)),
  );
  const savingsPct = Math.round((1 - 1 / speedupLow) * 100);

  return {
    healthScore,
    waste,
    speedupLow,
    speedupHigh,
    currentMs,
    projectedLowMs,
    projectedHighMs,
    projectedUtilLow,
    projectedUtilHigh,
    savingsPct,
    gpuActive,
    gpuActiveUnreliable,
    hasInputWait,
    launchBound,
    externalWorkflow,
  };
}

function workflowStep(summary: Summary, namePart: string) {
  const steps = summary.diagnosis.evidence.steps;
  if (!Array.isArray(steps)) return null;
  return steps.find((step) => {
    if (!step || typeof step !== "object") return false;
    const name = String((step as Record<string, unknown>).name ?? "").toLowerCase();
    return name.includes(namePart);
  }) as Record<string, unknown> | undefined ?? null;
}

function workflowEvidenceItems(summary: Summary) {
  const items: Array<{ label: string; value: string; action: string }> = [];
  const totalMs = numericValue(summary.run_summary.total_duration_ms);
  const retrieval = workflowStep(summary, "retrieve");
  const generation = workflowStep(summary, "model");
  const rerank = workflowStep(summary, "rerank");

  if (retrieval) {
    const duration = numericValue(retrieval.duration_ms);
    const calls = numericValue(retrieval.calls);
    items.push({
      label: "Retrieval fanout",
      value: calls !== null ? `${calls} calls` : formatDuration(duration),
      action: duration !== null && totalMs !== null
        ? `${pct((duration / totalMs) * 100)} of run time is spent retrieving context.`
        : "Retrieval dominates the suspected bottleneck list.",
    });
  }
  if (generation) {
    const tokens = numericValue(generation.input_tokens);
    const calls = numericValue(generation.calls);
    items.push({
      label: "Model context",
      value: tokens !== null ? `${Math.round(tokens / 1000)}k input tokens` : `${calls ?? "Multiple"} calls`,
      action: "Repeated context increases model latency and token cost.",
    });
  }
  if (rerank) {
    const calls = numericValue(rerank.calls);
    const duration = numericValue(rerank.duration_ms);
    items.push({
      label: "Reranking",
      value: calls !== null ? `${calls} calls` : formatDuration(duration),
      action: "Reranking once per retrieval hit should usually be batched.",
    });
  }
  if (totalMs !== null) {
    items.push({
      label: "Run duration",
      value: formatDuration(totalMs),
      action: "Use the next run to verify end-to-end latency improvement.",
    });
  }

  return items.slice(0, 4);
}

function measuredWorkflowConfidence(payload: ExternalRunPayload, primary: string | null) {
  if (!primary) {
    return {
      level: "low",
      score: 0,
      reason: "No suspected_bottlenecks were supplied.",
    };
  }

  const totalDuration = numericValue(payload.total_duration_ms);
  const normalizedPrimary = primary.toLowerCase();
  const steps = payload.steps ?? [];
  const measuredStep = steps.find((step) => {
    const name = String(step.name ?? "").toLowerCase();
    return (
      (normalizedPrimary.includes("retrieval") && name.includes("retrieve")) ||
      (normalizedPrimary.includes("context") && name.includes("retrieve")) ||
      (normalizedPrimary.includes("prompt") && name.includes("model")) ||
      (normalizedPrimary.includes("model") && name.includes("model")) ||
      (normalizedPrimary.includes("rerank") && name.includes("rerank")) ||
      (normalizedPrimary.includes("postprocess") && name.includes("postprocess"))
    );
  });
  const measuredDuration = measuredStep ? numericValue(measuredStep.duration_ms) : null;

  if (totalDuration !== null && measuredDuration !== null && totalDuration > 0) {
    const share = measuredDuration / totalDuration;
    const score = share >= 0.50 ? 0.92 : share >= 0.30 ? 0.84 : share >= 0.15 ? 0.72 : 0.60;
    const level = score >= 0.80 ? "high" : score >= 0.55 ? "medium" : "low";
    return {
      level,
      score,
      reason: `${displayText(measuredStep?.name)} accounts for ${pct(share * 100)} of measured run time.`,
    };
  }

  return {
    level: "medium",
    score: 0.7,
    reason: "Using suspected_bottlenecks supplied in the pasted run summary.",
  };
}

function evidenceItems(summary: Summary) {
  if (isExternalWorkflowSummary(summary)) {
    return workflowEvidenceItems(summary);
  }

  const items: Array<{ label: string; value: string; action: string }> = [];
  const metrics = reportMetrics(summary);
  const primaryEvidence = summary.bottlenecks[0]?.evidence ?? {};
  const dataWait = numericValue(primaryEvidence.avg_dataloader_fraction);
  const h2d = numericValue(primaryEvidence.avg_h2d_fraction);
  const sync = numericValue(primaryEvidence.avg_sync_fraction);
  const totalMs = numericValue(summary.run_summary.total_duration_ms);
  const kernelCount = numericValue(primaryEvidence.kernel_count_per_step)
    ?? numericValue(summary.run_summary.kernel_count_per_step);
  const medianKernelUs = numericValue(primaryEvidence.median_cuda_kernel_duration_us)
    ?? numericValue(summary.run_summary.median_cuda_kernel_duration_us);
  const smallKernelFraction = numericValue(primaryEvidence.small_kernel_fraction)
    ?? numericValue(summary.run_summary.small_kernel_fraction);

  if (metrics.gpuActive !== null) {
    items.push({
      label: metrics.launchBound ? "GPU activity" : "GPU active",
      value: metrics.launchBound && metrics.gpuActiveUnreliable
        ? "Bursty"
        : metrics.gpuActiveUnreliable ? "Very low / unreliable" : pct(metrics.gpuActive),
      action: metrics.launchBound && metrics.gpuActiveUnreliable
        ? "Sampling reports near-zero because the kernels are too short and bursty, not because no GPU work ran."
        : metrics.gpuActiveUnreliable
        ? "Profiler trace did not expose reliable CUDA busy time inside detected steps."
        : metrics.waste > 0
        ? `${pct(metrics.waste)} wait/idle time points to wasted accelerator spend.`
        : "GPU active estimate is saturated; trust the bottleneck-specific wait signal below.",
    });
  }
  if (metrics.launchBound && kernelCount !== null && kernelCount > 0) {
    items.push({
      label: "Kernel launches",
      value: `${Math.round(kernelCount)}/step`,
      action: "High launch count per step points to CPU launch overhead and fragmented GPU work.",
    });
  }
  if (metrics.launchBound && medianKernelUs !== null && medianKernelUs > 0) {
    items.push({
      label: "Short CUDA kernels",
      value: `${medianKernelUs.toFixed(medianKernelUs >= 10 ? 1 : 2)}us median`,
      action: smallKernelFraction !== null && smallKernelFraction > 0
        ? `${pct(smallKernelFraction * 100)} of measured kernels are under 10us.`
        : "Median kernel duration is very short, so launch overhead can dominate useful work.",
    });
  }
  if (dataWait !== null) {
    items.push({
      label: "Data wait",
      value: pct(dataWait * 100),
      action: "Likely DataLoader bottleneck; increase workers or prefetching.",
    });
  }
  if (h2d !== null) {
    items.push({
      label: "H2D transfer",
      value: pct(h2d * 100),
      action: "Host-to-device copies are visible; use pinned memory or overlap.",
    });
  }
  if (sync !== null) {
    items.push({
      label: "Sync wait",
      value: pct(sync * 100),
      action: "Host/device synchronization is stalling the step loop.",
    });
  }
  if (totalMs !== null) {
    items.push({
      label: "Run duration",
      value: formatDuration(totalMs),
      action: "Use the next run to verify end-to-end latency improvement.",
    });
  }
  if (items.length < 4) {
    items.push({
      label: "Pattern confidence",
      value: pct(summary.diagnosis.confidence.score * 100),
      action: displayText(summary.diagnosis.confidence.reason),
    });
  }
  if (items.length < 4 && summary.event_count > 0) {
    const lowStepCount = summary.step_count > 0 && summary.step_count < 20;
    items.push({
      label: "Trace evidence",
      value: lowStepCount ? "Limited" : `${summary.event_count} events`,
      action: lowStepCount
        ? `Only ${summary.step_count} steps captured. Enough to detect a likely pattern, not enough for stable ranking.`
        : "Enough signal for a ranked action plan; deeper trace remains optional.",
    });
  }

  return items.slice(0, 4);
}

function bottleneckLabel(key: unknown) {
  const normalizedKey = typeof key === "string" ? key : displayText(key);
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
  return normalizedKey ? (map[normalizedKey] ?? normalizedKey.replace(/_/g, " ")) : "No bottleneck detected";
}

function isExternalRunPayload(value: unknown): value is ExternalRunPayload {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = value as ExternalRunPayload;
  return Array.isArray(payload.steps) || Array.isArray(payload.suspected_bottlenecks);
}

function isProfilerTracePayload(value: unknown): value is ProfilerTracePayload {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = value as ProfilerTracePayload;
  return Array.isArray(payload.traceEvents);
}

function isRunMetadataPayload(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = value as Record<string, unknown>;
  return Array.isArray(payload.known_bottlenecks) || typeof payload.workload_name === "string";
}

function bundleFileName(file: File) {
  return file.webkitRelativePath || file.name;
}

function bundleStatusForFiles(files: File[], selectedFile: string | null): BundleStatus {
  const names = files.map(bundleFileName);
  const lowerNames = names.map((name) => name.toLowerCase());
  const hasSummary = lowerNames.some((name) => name.endsWith("summary.json"));
  const hasMetadata = lowerNames.some((name) => name.endsWith("metadata.json"));
  const hasProfilerTrace = lowerNames.some((name) => name.endsWith("profiler_trace.json"));
  const hasGpuMetrics = lowerNames.some((name) => name.endsWith("gpu_metrics.csv"));
  const hasRunConfig = lowerNames.some((name) => name.endsWith("run_config.yaml") || name.endsWith("run_config.yml"));
  const hasLogs = lowerNames.some((name) => name.endsWith("optional_logs.txt") || name.endsWith(".log"));

  return {
    fileNames: names,
    selectedFile,
    hasSummary,
    hasMetadata,
    hasProfilerTrace,
    hasGpuMetrics,
    hasRunConfig,
    hasLogs,
    limited: !hasSummary && (!hasRunConfig || !(hasMetadata && (hasProfilerTrace || hasGpuMetrics))),
  };
}

// Detects the multi-scope output of summarize_run_with_steady_state (frx collect).
function isFrxCliSummary(payload: unknown): boolean {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  const r = payload as Record<string, unknown>;
  const run = r.run as Record<string, unknown> | undefined;
  const ss = r.steady_state as Record<string, unknown> | undefined;
  return !!(run?.diagnosis || ss?.diagnosis);
}

// Extract the best scope from a CLI multi-scope summary as a flat Summary object.
function frxCliSummaryToSummary(payload: Record<string, unknown>): Summary {
  const scope = ((payload.steady_state ?? payload.run) as Record<string, unknown>);
  return {
    event_count: typeof payload.event_count === "number" ? payload.event_count : (scope.event_count as number ?? 0),
    step_count: scope.step_count as number ?? 0,
    run_summary: scope.run_summary as Record<string, unknown> ?? {},
    bottlenecks: scope.bottlenecks as Summary["bottlenecks"] ?? [],
    diagnosis: scope.diagnosis as Diagnosis,
    scope: scope.scope as Summary["scope"] ?? { name: "steady_state", step_ids: [] },
  };
}

function scoreBundleAnalysisPayload(file: File, payload: unknown) {
  const name = bundleFileName(file).toLowerCase();
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    const record = payload as Record<string, unknown>;
    if (isFrxCliSummary(record)) return 130;  // frx collect derived/summary.json — highest priority
    if (record.diagnosis) return 120;
    if (Array.isArray(record.events)) return 110;
    if (Array.isArray(record.traceEvents)) return 100;
    if (Array.isArray(record.steps) || Array.isArray(record.suspected_bottlenecks)) return 90;
    if (Array.isArray(record.known_bottlenecks)) return 40;
  }
  if (Array.isArray(payload)) return 110;
  if (name.endsWith("profiler_trace.json")) return 80;
  if (name.endsWith("metadata.json")) return 30;
  if (name.endsWith(".json")) return 20;
  return 0;
}

async function bestAnalyzableBundleFile(files: File[]) {
  const candidates: Array<{ file: File; text: string; score: number }> = [];
  for (const file of files.filter((item) => bundleFileName(item).toLowerCase().endsWith(".json"))) {
    const text = await file.text();
    try {
      const payload = JSON.parse(text);
      const score = scoreBundleAnalysisPayload(file, payload);
      if (score > 0) candidates.push({ file, text, score });
    } catch {
      // Try the next JSON file in the bundle.
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  return candidates[0] ?? null;
}

function isStepEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  const cat = (event.cat ?? "").toLowerCase();
  return (
    event.ph === "X" &&
    typeof event.ts === "number" &&
    typeof event.dur === "number" &&
    (
      name.includes("train_step") ||
      name.includes("profilerstep") ||
      name.includes("training_step") ||
      (cat.includes("user_annotation") && name.includes("step"))
    )
  );
}

function eventWithinStep(event: ProfilerTraceEvent, step: ProfilerTraceEvent) {
  if (typeof event.ts !== "number" || typeof event.dur !== "number") return false;
  if (typeof step.ts !== "number" || typeof step.dur !== "number") return false;
  const eventMidpoint = event.ts + event.dur / 2;
  return eventMidpoint >= step.ts && eventMidpoint <= step.ts + step.dur;
}

function profilerDurationNs(event: ProfilerTraceEvent) {
  return Math.max(0, Math.round((event.dur ?? 0) * 1000));
}

function phaseNameForProfilerEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  if (name.includes("forward")) return "forward";
  if (name.includes("backward") || name.includes("autograd")) return "backward";
  if (name.includes("optimizer") || name.includes("adam") || name.includes("sgd")) return "optimizer";
  return null;
}

function isDataLoaderProfilerEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  return name.includes("dataloader") || name.includes("enumerate(data") || name.includes("__next__");
}

function isH2DProfilerEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  return name.includes("memcpy") && (name.includes("h2d") || name.includes("host to device") || name.includes("htod"));
}

function isSyncProfilerEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  return name.includes("synchronize") || name.includes("cuda_stream_synchronize") || name.includes("cuda_device_synchronize");
}

function isCudaKernelProfilerEvent(event: ProfilerTraceEvent) {
  const name = (event.name ?? "").toLowerCase();
  const cat = (event.cat ?? "").toLowerCase();
  return cat.includes("kernel") || name.includes("cuda_time") || name.includes("cuda kernel");
}

function unionDurationUs(events: ProfilerTraceEvent[]) {
  const intervals = events
    .filter((event) => typeof event.ts === "number" && typeof event.dur === "number" && event.dur > 0)
    .map((event) => [event.ts as number, (event.ts as number) + (event.dur as number)] as const)
    .sort((a, b) => a[0] - b[0]);
  let total = 0;
  let currentStart: number | null = null;
  let currentEnd = 0;

  for (const [start, end] of intervals) {
    if (currentStart === null) {
      currentStart = start;
      currentEnd = end;
      continue;
    }
    if (start <= currentEnd) {
      currentEnd = Math.max(currentEnd, end);
      continue;
    }
    total += currentEnd - currentStart;
    currentStart = start;
    currentEnd = end;
  }

  if (currentStart !== null) total += currentEnd - currentStart;
  return total;
}

function profilerTraceToEvents(payload: ProfilerTracePayload) {
  const traceEvents = payload.traceEvents.filter((event) => (
    event.ph === "X" &&
    typeof event.ts === "number" &&
    typeof event.dur === "number" &&
    event.dur > 0
  ));
  let steps = traceEvents.filter(isStepEvent).sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));

  if (steps.length === 0 && traceEvents.length > 0) {
    const firstTs = Math.min(...traceEvents.map((event) => event.ts ?? 0));
    const lastEnd = Math.max(...traceEvents.map((event) => (event.ts ?? 0) + (event.dur ?? 0)));
    steps = [{
      ph: "X",
      cat: "synthetic",
      name: "profiler_trace",
      ts: firstTs,
      dur: Math.max(0, lastEnd - firstTs),
      args: {},
    }];
  }

  const converted: Array<Record<string, unknown>> = [];

  steps.forEach((step, index) => {
    const stepId = index + 1;
    const stepDurationNs = profilerDurationNs(step);
    const childEvents = traceEvents.filter((event) => event !== step && eventWithinStep(event, step));

    converted.push({ event_type: "step_start", step_id: stepId, payload: { step_kind: "train" } });
    converted.push({
      event_type: "profiler_window",
      step_id: stepId,
      payload: { window_state: "exported", source: "traceEvents", trace_id: payload.trace_id },
    });

    const kernelBusyNs = Math.round(unionDurationUs(childEvents.filter(isCudaKernelProfilerEvent)) * 1000);
    let memoryUsedBytes = 0;
    let memoryTotalBytes = 0;

    for (const event of childEvents) {
      const durationNs = profilerDurationNs(event);
      const args = event.args ?? {};
      const deviceMemory = numericValue(args["Device Memory Usage"]);
      const totalReserved = numericValue(args["Total Reserved"]);
      if (deviceMemory !== null) memoryUsedBytes = Math.max(memoryUsedBytes, deviceMemory);
      if (totalReserved !== null) memoryTotalBytes = Math.max(memoryTotalBytes, totalReserved);

      if (isDataLoaderProfilerEvent(event)) {
        converted.push({
          event_type: "dataloader_span",
          step_id: stepId,
          duration_ns: durationNs,
          payload: { stage: "next", source_name: event.name },
        });
        continue;
      }

      if (isH2DProfilerEvent(event)) {
        converted.push({
          event_type: "memcpy_span",
          step_id: stepId,
          duration_ns: durationNs,
          payload: { copy_kind: "h2d", source_name: event.name },
        });
        continue;
      }

      if (isSyncProfilerEvent(event)) {
        converted.push({
          event_type: "sync_wait",
          step_id: stepId,
          duration_ns: durationNs,
          payload: { wait_kind: "profiler_sync", source_name: event.name },
        });
        continue;
      }

      const phaseName = phaseNameForProfilerEvent(event);
      if (phaseName) {
        converted.push({
          event_type: "phase_span",
          step_id: stepId,
          duration_ns: durationNs,
          payload: { phase_name: phaseName, source_name: event.name },
        });
      }
    }

    if (!converted.some((event) => event["event_type"] === "phase_span" && event["step_id"] === stepId) && kernelBusyNs > 0) {
      converted.push({
        event_type: "phase_span",
        step_id: stepId,
        duration_ns: Math.min(kernelBusyNs, stepDurationNs),
        payload: { phase_name: "forward", source_name: "cuda_kernel_busy_estimate" },
      });
    }

    const gpuUtil = stepDurationNs > 0 ? Math.min(100, Math.round((kernelBusyNs / stepDurationNs) * 100)) : 0;
    converted.unshift({
      event_type: "gpu_sample",
      payload: {
        utilization_gpu_pct: gpuUtil,
        utilization_mem_pct: memoryTotalBytes > 0 ? Math.round((memoryUsedBytes / memoryTotalBytes) * 100) : 0,
        memory_used_bytes: memoryUsedBytes,
        memory_total_bytes: memoryTotalBytes || 1,
      },
    });
    converted.push({
      event_type: "step_end",
      step_id: stepId,
      duration_ns: stepDurationNs,
      payload: { step_kind: "train", status: "ok", source_name: step.name },
    });
  });

  return converted;
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

function builtInFix(
  id: string,
  title: string,
  priority: Recommendation["priority"],
  why: string,
  actions: string[],
  index: number,
): Recommendation {
  return {
    id,
    title,
    priority,
    score: Math.max(0.4, 0.88 - index * 0.06),
    confidence: 0.75,
    expected_impact: priority === "high" ? "high" : "medium",
    effort: index < 3 ? "low" : "medium",
    category: "fallback",
    why,
    actions,
    validation: [
      "Re-run the same workload and compare step time.",
      "Confirm the primary bottleneck score drops after the change.",
    ],
    risks: ["This fallback fix is generated from the primary bottleneck label."],
    triggered_by: "primary_bottleneck_fallback",
  };
}

function fallbackFixesForPrimary(primary: string | null): Recommendation[] {
  const normalized = slugifyBottleneck(displayText(primary));
  if (normalized.includes("input_bound") || normalized.includes("input_pipeline")) {
    return [
      builtInFix(
        "rec_input_num_workers_fallback",
        "Increase DataLoader workers",
        "high",
        "Parallelizes sample loading so the GPU waits less between batches.",
        ["Benchmark num_workers at 4, 8, and 12.", "Choose the lowest setting that reduces DataLoader wait without CPU contention."],
        0,
      ),
      builtInFix(
        "rec_input_pinned_memory_fallback",
        "Enable pin_memory=True",
        "medium",
        "Speeds host-to-device batch transfer.",
        ["Set pin_memory=True on the PyTorch DataLoader.", "Use non_blocking=True when moving batches to the GPU."],
        1,
      ),
      builtInFix(
        "rec_input_prefetch_factor_fallback",
        "Increase prefetch_factor",
        "medium",
        "Keeps future batches ready before the GPU asks for them.",
        ["Set prefetch_factor=2 or 4 when num_workers > 0.", "Watch host RAM while increasing prefetch depth."],
        2,
      ),
      builtInFix(
        "rec_input_move_transforms_fallback",
        "Move CPU-heavy transforms off the hot path",
        "medium",
        "Reduces per-sample CPU preprocessing delay.",
        ["Profile dataset __getitem__.", "Cache, precompute, or move expensive transforms out of the critical path."],
        3,
      ),
    ];
  }
  return [];
}

function isSearchFanoutLabel(normalized: string) {
  return (
    (normalized.includes("search") || normalized.includes("retrieval")) &&
    (
      normalized.includes("fan") ||
      normalized.includes("duplicate") ||
      normalized.includes("near-duplicate") ||
      normalized.includes("similar") ||
      normalized.includes("overlap")
    )
  );
}

function whyForExternalPrimary(primary: string | null) {
  if (!primary) return ["No suspected bottlenecks were supplied in this run summary."];

  const normalized = primary.toLowerCase();
  if (isSearchFanoutLabel(normalized)) {
    return [
      "Search appears to be called multiple times for semantically similar queries, which likely adds avoidable latency before generation begins.",
    ];
  }
  if (normalized.includes("prompt") || normalized.includes("context") || normalized.includes("token")) {
    return [
      "Model calls appear to receive repeated or oversized context, which likely adds latency and token cost without adding new signal.",
    ];
  }
  if (normalized.includes("postprocess") || normalized.includes("format")) {
    return [
      "Response formatting appears slow enough to affect user-visible latency, so it needs its own timing breakdown.",
    ];
  }

  return [
    "The pasted summary reports this as the leading suspected bottleneck, but more targeted telemetry would improve fix confidence.",
  ];
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
    if (normalized.includes("rerank")) {
      addFix(
        "Batch rerank_results calls",
        "high",
        "Reranking once per retrieval hit adds avoidable latency that can usually be collapsed into one batch.",
        [
          "Collect retrieval hits first, then call rerank_results once per query group.",
          "Pass the candidate list as a batch instead of reranking each hit independently.",
          "Track rerank call count and total rerank latency in the next run.",
        ],
      );
      continue;
    }
    if (normalized.includes("duplicate") && normalized.includes("model")) {
      addFix(
        "Skip duplicate model calls within the same run",
        "medium",
        "The trace shows a likely repeated model call with similar token counts and outputs, which can be eliminated before paying model latency.",
        [
          "Hash the normalized prompt, retrieved context IDs, and generation settings before each model call.",
          "If the hash was already seen in this run, reuse the previous response instead of calling the model again.",
          "Log duplicate model-call skips and latency saved per run.",
        ],
      );
      continue;
    }
    if (isSearchFanoutLabel(normalized)) {
      addFix(
        "Merge semantically duplicate search queries",
        "high",
        "Equivalent search intents are paying fan-out latency without adding meaningfully different evidence.",
        [
          "Normalize query text by lowercasing, trimming, and removing low-signal word-order differences.",
          "Cluster near-duplicate queries before issuing search calls.",
          "Issue one canonical query for examples like wireless noise cancelling headphones and noise cancelling wireless headphones.",
        ],
      );
      addFix(
        "Cache repeated or near-duplicate search results",
        "high",
        "Overlapping search intent can reuse the same result set instead of waiting on repeated search calls.",
        [
          "Key the cache by canonical query text plus search settings.",
          "Use a run-scoped cache so repeated intents reuse the first result.",
          "Log cache hit rate, skipped search calls, and search latency saved.",
        ],
      );
    } else if (normalized.includes("retrieval")) {
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
      if (normalized.includes("cache")) {
        addFix(
          "Enable prompt or context caching",
          "high",
          "Full context is being re-sent across model calls, which pays repeated token processing cost.",
          [
            "Cache stable system instructions and retrieved context across related model calls.",
            "Send only the delta when the next call reuses the same retrieved context.",
            "Track cached input tokens and model-call latency after enabling caching.",
          ],
        );
      }
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
  const confidence = measuredWorkflowConfidence(payload, primary);

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
      confidence,
      evidence: {
        total_duration_ms: totalDuration,
        steps,
        events,
      },
      why: primary
        ? whyForExternalPrimary(primary)
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

function metadataToExternalRun(payload: Record<string, unknown>): ExternalRunPayload {
  const known = Array.isArray(payload.known_bottlenecks)
    ? payload.known_bottlenecks.map(displayText).filter(Boolean)
    : [];
  const steps = [];
  const elapsedSeconds = numericValue(payload.elapsed_seconds);
  if (elapsedSeconds !== null) {
    steps.push({
      name: displayText(payload.workload_name) || "run",
      duration_ms: elapsedSeconds * 1000,
      calls: numericValue(payload.steps) ?? 1,
    });
  }

  return {
    run_id: displayText(payload.workload_name) || "metadata-only-run",
    status: "metadata_only",
    total_duration_ms: elapsedSeconds !== null ? elapsedSeconds * 1000 : undefined,
    steps,
    events: [],
    suspected_bottlenecks: known.length > 0
      ? known
      : ["Metadata-only upload; collect profiler_trace.json for bottleneck timing."],
  };
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RecCard({
  rec,
  rank,
  feedback,
  onFeedback,
}: {
  rec: Recommendation;
  rank: number;
  feedback: Record<string, string>;
  onFeedback: (recId: string, outcome: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const given = feedback[rec.id];

  return (
    <div className="overflow-hidden rounded-lg border border-white/10 bg-white/[0.035]">
      <div className="flex items-start justify-between gap-4 px-4 py-4">
        <div className="flex min-w-0 items-start gap-3">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-cyan-400/25 bg-cyan-400/10 text-sm font-semibold text-cyan-100">
            {rank}
          </span>
          <span
            className={`mt-0.5 shrink-0 rounded-full border px-2 py-0.5 text-[0.6rem] font-semibold uppercase tracking-[0.18em] ${priorityStyle(rec.priority)}`}
          >
            Priority: {rec.priority}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-white">{displayText(rec.title)}</p>
            <p className="mt-0.5 text-xs leading-5 text-slate-400 line-clamp-2">{displayText(rec.why)}</p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
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

      {expanded && (
        <div className="border-t border-white/10 px-5 py-4 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Section title="Actions">
              <ol className="list-decimal list-inside space-y-1">
                {rec.actions.map((a, i) => (
                  <li key={i} className="text-xs text-slate-300">{displayText(a)}</li>
                ))}
              </ol>
            </Section>
            <Section title="How to verify">
              <ul className="space-y-1">
                {rec.validation.map((v, i) => (
                  <li key={i} className="flex gap-2 text-xs text-slate-300">
                    <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-emerald-500" />
                    {displayText(v)}
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
                    {displayText(r)}
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

function SectionHeader({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div>
      <p className="text-[0.6rem] font-semibold uppercase tracking-[0.2em] text-slate-500">{eyebrow}</p>
      <h3 className="mt-1 text-sm font-semibold text-white">{title}</h3>
    </div>
  );
}

function VerdictStat({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <div className="flex items-center gap-2 text-slate-500">
        <Icon size={14} />
        <span className="text-[0.6rem] font-semibold uppercase tracking-[0.16em]">{label}</span>
      </div>
      <p className={`mt-2 text-lg font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function ProjectionStat({ label, current, projected }: { label: string; current: string; projected: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-4">
      <p className="text-xs font-medium text-slate-400">{label}</p>
      <div className="mt-3 flex items-baseline justify-between gap-3">
        <span className="text-sm text-slate-500">{current}</span>
        <span className="text-lg font-semibold text-emerald-300">{projected}</span>
      </div>
    </div>
  );
}

function ComparisonRow({ metric, runA, runB }: { metric: string; runA: string; runB: string }) {
  return (
    <div className="grid grid-cols-[1fr_0.8fr_0.8fr] gap-3 border-b border-white/10 px-3 py-3 text-xs last:border-b-0">
      <span className="text-slate-400">{metric}</span>
      <span className="text-slate-200">{runA}</span>
      <span className="font-medium text-emerald-300">{runB}</span>
    </div>
  );
}

function BundlePill({ label, active }: { label: string; active: boolean }) {
  return (
    <span className={`rounded-full border px-2.5 py-1 font-semibold uppercase tracking-[0.12em] ${
      active
        ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-200"
        : "border-white/10 bg-white/[0.03] text-slate-500"
    }`}>
      {label}
    </span>
  );
}

function ReportBody({
  summary,
  feedback,
  onFeedback,
}: {
  summary: Summary;
  feedback: Record<string, string>;
  onFeedback: (recId: string, outcome: string) => void;
}) {
  const { diagnosis } = summary;
  const [showTrace, setShowTrace] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const metrics = reportMetrics(summary);
  const evidence = evidenceItems(summary);
  const recommendations = diagnosis.recommendations.length > 0
    ? diagnosis.recommendations
    : fallbackFixesForPrimary(diagnosis.primary_bottleneck);
  const topRecs = recommendations.slice(0, 5);
  const secondary = diagnosis.secondary_bottlenecks.slice(0, 2);
  const sampleSizeLow = !metrics.externalWorkflow && summary.step_count > 0 && summary.step_count < 20;
  const displayedConfidenceLevel = sampleSizeLow && diagnosis.confidence.level === "high"
    ? "medium"
    : diagnosis.confidence.level;
  const displayedConfidenceScore = sampleSizeLow
    ? Math.min(diagnosis.confidence.score, 0.7)
    : diagnosis.confidence.score;
  const traceBars = evidence.length > 0 ? evidence : [
    { label: "Trace", value: `${summary.event_count} events`, action: "Raw details are available below." },
  ];
  const incomplete = isAnalysisIncomplete(summary);

  if (incomplete) {
    return (
      <div className="space-y-6">
        <section className="rounded-xl border border-yellow-400/25 bg-yellow-400/10 p-5">
          <div className="flex items-start gap-3">
            <AlertCircle size={20} className="mt-1 shrink-0 text-yellow-200" />
            <div className="min-w-0">
              <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-yellow-200/80">
                Analysis incomplete
              </p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">
                Trace data is missing or not measurable
              </h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-yellow-50/85">
                {incompleteAnalysisReason(summary)}
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
          <SectionHeader eyebrow="Required evidence" title="What to collect next" />
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-white/10 bg-black/20 p-4">
              <p className="text-sm font-medium text-slate-200">Profiler trace</p>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                Pass <code className="text-slate-300">--artifact-dir</code> for the workload directory that contains <code className="text-slate-300">profiler_trace.json</code>.
              </p>
            </div>
            <div className="rounded-lg border border-white/10 bg-black/20 p-4">
              <p className="text-sm font-medium text-slate-200">Step timing</p>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                Keep the top-level <code className="text-slate-300">record_function</code> wrapper or <code className="text-slate-300">ProfilerStep#N</code> events in the trace.
              </p>
            </div>
            <div className="rounded-lg border border-white/10 bg-black/20 p-4">
              <p className="text-sm font-medium text-slate-200">GPU samples</p>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                Include <code className="text-slate-300">gpu_metrics.csv</code> so low utilization can be distinguished from missing telemetry.
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-white/10 bg-white/[0.03]">
          <button
            onClick={() => setShowRaw((value) => !value)}
            className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
          >
            <SectionHeader eyebrow="Raw details" title="Power-user data" />
            {showRaw ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
          </button>
          {showRaw && (
            <div className="border-t border-white/10 px-5 py-4">
              <pre className="max-h-80 overflow-auto rounded-lg bg-black/30 p-4 text-xs leading-5 text-slate-400">
                {JSON.stringify({
                  run_summary: summary.run_summary,
                  bottlenecks: summary.bottlenecks,
                  diagnosis: summary.diagnosis,
                }, null, 2)}
              </pre>
            </div>
          )}
        </section>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-white/10 bg-white/[0.04] p-5">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div className="min-w-0">
            <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-slate-500">
              Diagnosis verdict
            </p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">
              {bottleneckLabel(diagnosis.user_facing_bottleneck ?? diagnosis.primary_bottleneck)}
            </h2>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1 text-slate-300">
                Primary: {bottleneckLabel(diagnosis.user_facing_bottleneck ?? diagnosis.primary_bottleneck)}
              </span>
              {secondary.map((item) => (
                <span key={displayText(item)} className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1 text-slate-400">
                  Also: {bottleneckLabel(item)}
                </span>
              ))}
            </div>
          </div>
          <div className="grid min-w-[15rem] grid-cols-2 gap-3 sm:min-w-[22rem]">
            <VerdictStat
              icon={Gauge}
              label={metrics.externalWorkflow ? "Workflow health" : "GPU health"}
              value={`${metrics.healthScore}/100`}
              tone={healthColor(metrics.healthScore)}
            />
            <VerdictStat
              icon={Target}
              label={metrics.externalWorkflow ? "Latency waste" : metrics.launchBound ? "Launch overhead waste" : "Wait waste"}
              value={metrics.launchBound ? "High" : pct(metrics.waste)}
              tone="text-red-300"
            />
            <VerdictStat icon={TrendingUp} label="Speedup" value={`+${metrics.speedupLow.toFixed(1)}x-${metrics.speedupHigh.toFixed(1)}x`} tone="text-emerald-300" />
            <VerdictStat icon={Activity} label="Confidence" value={`${displayedConfidenceLevel} ${pct(displayedConfidenceScore * 100)}`} tone={confidenceStyle(displayedConfidenceLevel)} />
          </div>
        </div>
        {sampleSizeLow && (
          <div className="mt-4 rounded-lg border border-yellow-400/20 bg-yellow-400/10 px-4 py-3 text-sm leading-6 text-yellow-100">
            Low sample size: only {summary.step_count} steps captured. Treat this as a directional diagnosis; collect 20-50 steps for a reliable ranking.
          </div>
        )}
      </section>

      <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
        <SectionHeader eyebrow="Observed evidence" title="What's going wrong" />
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {evidence.map((item) => (
            <div key={item.label} className="rounded-lg border border-white/10 bg-black/20 p-4">
              <div className="flex items-baseline justify-between gap-3">
                <p className="text-sm font-medium text-slate-200">{item.label}</p>
                <p className="text-lg font-semibold text-white">{item.value}</p>
              </div>
              <p className="mt-2 text-xs leading-5 text-slate-400">{item.action}</p>
            </div>
          ))}
        </div>
        {diagnosis.why.length > 0 && (
          <p className="mt-4 text-sm leading-6 text-slate-300">{displayText(diagnosis.why[0])}</p>
        )}
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between gap-3">
          <SectionHeader eyebrow="Action plan" title="Recommended fixes" />
          <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[0.65rem] text-slate-400">
            {topRecs.length} fix{topRecs.length !== 1 ? "es" : ""}
          </span>
        </div>
        <div className="space-y-2">
          {topRecs.map((rec, index) => (
            <RecCard
              key={rec.id}
              rank={index + 1}
              rec={rec}
              feedback={feedback}
              onFeedback={onFeedback}
            />
          ))}
          {topRecs.length === 0 && (
            <p className="rounded-lg border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-400">
              No recommendations generated for this profile.
            </p>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
        <SectionHeader eyebrow="Projected result" title="What happens if you fix this" />
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <ProjectionStat
            label={metrics.externalWorkflow
              ? "Workflow latency"
              : metrics.launchBound ? "GPU utilization"
              : metrics.gpuActive !== null && !metrics.gpuActiveUnreliable ? "GPU utilization" : "GPU activity"}
            current={metrics.externalWorkflow
              ? "Baseline"
              : metrics.launchBound ? "Bursty; not reliably sampled"
              : metrics.gpuActive !== null && !metrics.gpuActiveUnreliable ? pct(metrics.gpuActive) : "Not reliably measured"}
            projected={metrics.externalWorkflow
              ? "Expected to improve after top fixes"
              : metrics.launchBound
              ? "Fewer launches, lower step time"
              : metrics.projectedUtilLow !== null && metrics.projectedUtilHigh !== null
              ? `${metrics.projectedUtilLow}-${metrics.projectedUtilHigh}%`
              : "Expected to improve after reducing DataLoader wait"}
          />
          <ProjectionStat
            label="Step time"
            current={formatDuration(metrics.currentMs)}
            projected={metrics.projectedLowMs !== null && metrics.projectedHighMs !== null
              ? `${formatDuration(metrics.projectedLowMs)}-${formatDuration(metrics.projectedHighMs)}`
              : "Measure after fixes"}
          />
          <ProjectionStat label="Estimated cost savings" current="Baseline" projected={`~${metrics.savingsPct}%`} />
        </div>
      </section>

      <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
        <SectionHeader eyebrow="Run comparison" title="Current vs after top fixes" />
        <div className="mt-4 overflow-hidden rounded-lg border border-white/10">
          <ComparisonRow
            metric={metrics.externalWorkflow ? "Workflow latency" : "GPU activity"}
            runA={metrics.externalWorkflow
              ? "Baseline"
              : metrics.launchBound ? "Bursty; not reliably sampled"
              : metrics.gpuActive !== null && !metrics.gpuActiveUnreliable ? pct(metrics.gpuActive) : "Not reliably measured"}
            runB={metrics.externalWorkflow
              ? "Expected to improve"
              : metrics.launchBound
              ? "Fewer launches, lower step time"
              : metrics.projectedUtilLow !== null && metrics.projectedUtilHigh !== null ? `${metrics.projectedUtilLow}-${metrics.projectedUtilHigh}%` : "Expected to improve"}
          />
          <ComparisonRow metric="Step time" runA={formatDuration(metrics.currentMs)} runB={metrics.projectedLowMs !== null && metrics.projectedHighMs !== null ? `${formatDuration(metrics.projectedLowMs)}-${formatDuration(metrics.projectedHighMs)}` : "Re-run to confirm"} />
          <ComparisonRow metric="Primary bottleneck" runA={bottleneckLabel(diagnosis.user_facing_bottleneck ?? diagnosis.primary_bottleneck)} runB="Reduced or cleared" />
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-400">
          Biggest expected improvement: {topRecs[0] ? displayText(topRecs[0].title) : "apply the top-ranked recommendation and re-run the analyzer"}.
        </p>
      </section>

      <section className="rounded-xl border border-white/10 bg-white/[0.03]">
        <button
          onClick={() => setShowTrace((value) => !value)}
          className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
        >
          <SectionHeader eyebrow="Trace view" title="Timeline summary" />
          {showTrace ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
        </button>
        {showTrace && (
          <div className="border-t border-white/10 px-5 py-4">
            <div className="space-y-3">
              {traceBars.map((item, index) => (
                <div key={`${item.label}-${index}`}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="text-slate-300">{item.label}</span>
                    <span className="text-slate-500">{item.value}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-cyan-300"
                      style={{ width: `${Math.max(16, Math.min(100, parseFloat(item.value) || 45 + index * 12))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="rounded-xl border border-white/10 bg-white/[0.03]">
        <button
          onClick={() => setShowRaw((value) => !value)}
          className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
        >
          <SectionHeader eyebrow="Raw details" title="Power-user data" />
          {showRaw ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
        </button>
        {showRaw && (
          <div className="border-t border-white/10 px-5 py-4">
            <pre className="max-h-80 overflow-auto rounded-lg bg-black/30 p-4 text-xs leading-5 text-slate-400">
              {JSON.stringify({
                run_summary: summary.run_summary,
                bottlenecks: summary.bottlenecks,
                evidence: diagnosis.evidence,
              }, null, 2)}
            </pre>
          </div>
        )}
      </section>
    </div>
  );
}

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

  return <ReportBody summary={summary} feedback={feedback} onFeedback={handleFeedback} />;

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
              {bottleneckLabel(diagnosis.user_facing_bottleneck ?? diagnosis.primary_bottleneck)}
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
                  {displayText(bullet)}
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
              {diagnosis.secondary_bottlenecks.map((bottleneck, index) => (
                <li key={`${displayText(bottleneck)}-${index}`} className="flex gap-2 text-sm text-slate-300">
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
              {recs.map((rec, index) => (
                <RecCard
                  key={rec.id}
                  rank={index + 1}
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
            {unbundled.map((rec, index) => (
              <RecCard
                key={rec.id}
                rank={index + 1}
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
  const [bundleStatus, setBundleStatus] = useState<BundleStatus | null>(null);

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

      // frx collect derived/summary.json — multi-scope structure { run, steady_state }
      if (isFrxCliSummary(parsed)) {
        setSummary(frxCliSummaryToSummary(parsed));
        setLoading(false);
        return;
      }

      // Flat pre-analyzed summary with a top-level diagnosis key
      if (parsed.diagnosis) {
        setSummary(parsed as Summary);
        setLoading(false);
        return;
      }

      if (isProfilerTracePayload(parsed)) {
        const events = profilerTraceToEvents(parsed);
        if (events.length === 0) {
          throw new Error("Profiler trace did not contain duration events that can be analyzed.");
        }
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ events, environment: { framework: "pytorch" } }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(displayText(err.detail ?? err) || "API error");
        }
        const data = await res.json() as Summary;
        setSummary(data);
        setLoading(false);
        return;
      }

      if (isExternalRunPayload(parsed)) {
        setSummary(externalRunToSummary(parsed));
        setLoading(false);
        return;
      }

      if (isRunMetadataPayload(parsed)) {
        setSummary(externalRunToSummary(metadataToExternalRun(parsed)));
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
        throw new Error(displayText(err.detail ?? err) || "API error");
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
    setBundleStatus(null);
    parseAndDisplay(json);
  }

  async function handleBundleUpload(filesList: FileList | null) {
    const files = Array.from(filesList ?? []);
    if (files.length === 0) return;

    setError(null);
    setSummary(null);
    setRaw("");

    const best = await bestAnalyzableBundleFile(files);
    const status = bundleStatusForFiles(files, best ? bundleFileName(best.file) : null);
    setBundleStatus(status);

    if (!best) {
      setError("No analyzable JSON file found. Include derived/summary.json, profiler_trace.json, or an events JSON file.");
      return;
    }

    // If the best file is a frx CLI summary, display the report immediately — no button click needed.
    try {
      const parsed = JSON.parse(best.text);
      if (isFrxCliSummary(parsed)) {
        setSummary(frxCliSummaryToSummary(parsed));
        return;
      }
      if (parsed.diagnosis) {
        setSummary(parsed as Summary);
        return;
      }
    } catch {
      // Not JSON or not pre-analyzed — fall through to textarea for manual analysis.
    }

    setRaw(best.text);
  }

  return (
    <div className="rounded-xl border border-white/10 bg-[#0a0f1c]/80 p-4 shadow-2xl shadow-black/30 backdrop-blur sm:p-6">
      <div>
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
          <div>
            <p className="text-sm font-semibold text-white">Analyzer workspace</p>
            <p className="mt-1 text-xs text-slate-500">
              Import a <code className="text-slate-400">summary.json</code> from your run, drop the full <code className="text-slate-400">frx collect</code> folder, or paste JSON directly.
            </p>
          </div>
          <div className="flex rounded-md border border-white/10 bg-white/[0.03] p-1 text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-slate-500">
            <span className="rounded bg-cyan-400/10 px-2.5 py-1 text-cyan-200">JSON</span>
            <span className="px-2.5 py-1">Report</span>
          </div>
        </div>
        {!summary && (
          <div className="space-y-4">
            <div className="rounded-lg border border-white/10 bg-white/[0.03] p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-white">Import your run output</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    Upload the folder from <code className="text-slate-400">frx collect</code>. The analyzer auto-selects the best file — <code className="text-slate-400">summary.json</code> loads instantly with no extra steps.
                  </p>
                </div>
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/10">
                  <FolderUp size={15} />
                  Upload bundle
                  <input
                    type="file"
                    multiple
                    accept=".json,.csv,.yaml,.yml,.txt,.log"
                    className="hidden"
                    onChange={(event) => void handleBundleUpload(event.target.files)}
                    {...({ webkitdirectory: "true", directory: "true" } as Record<string, string>)}
                  />
                </label>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                {([
                  { name: "summary.json",        desc: "Main diagnosis",          highlight: true  },
                  { name: "metadata.json",        desc: "Run & hardware info",     highlight: false },
                  { name: "profiler_trace.json",  desc: "Kernel-level timing",     highlight: false },
                  { name: "gpu_metrics.csv",      desc: "Per-step GPU counters",   highlight: false },
                  { name: "run_config.yaml",      desc: "Hyperparameters",         highlight: false },
                  { name: "*.log",                desc: "Training logs (optional)",highlight: false },
                ] as const).map(({ name, desc, highlight }) => (
                  <div
                    key={name}
                    className={`flex items-start gap-2 rounded-md px-2.5 py-2 ${
                      highlight
                        ? "border border-cyan-400/20 bg-cyan-400/5"
                        : "border border-white/5 bg-white/[0.02]"
                    }`}
                  >
                    <span className={`mt-[3px] h-1.5 w-1.5 shrink-0 rounded-full ${highlight ? "bg-cyan-400" : "bg-white/15"}`} />
                    <div className="min-w-0">
                      <code className={`block truncate text-[0.65rem] font-medium leading-tight ${highlight ? "text-cyan-200" : "text-slate-400"}`}>
                        {name}
                      </code>
                      <span className="mt-0.5 block text-[0.6rem] leading-tight text-slate-600">{desc}</span>
                    </div>
                  </div>
                ))}
              </div>

              {bundleStatus && (
                <div className="mt-4 rounded-lg border border-white/10 bg-black/20 p-3">
                  <div className="flex flex-wrap gap-2 text-[0.65rem]">
                    <BundlePill label="summary.json" active={bundleStatus.hasSummary} />
                    <BundlePill label="metadata.json" active={bundleStatus.hasMetadata} />
                    <BundlePill label="profiler_trace.json" active={bundleStatus.hasProfilerTrace} />
                    <BundlePill label="gpu_metrics.csv" active={bundleStatus.hasGpuMetrics} />
                    <BundlePill label="run_config.yaml" active={bundleStatus.hasRunConfig} />
                    <BundlePill label="logs" active={bundleStatus.hasLogs} />
                  </div>
                  <p className="mt-3 text-xs text-slate-400">
                    Loaded {bundleStatus.fileNames.length} file{bundleStatus.fileNames.length !== 1 ? "s" : ""}.
                    {bundleStatus.selectedFile ? ` Analyzing ${bundleStatus.selectedFile}.` : ""}
                  </p>
                  {bundleStatus.limited && (
                    <div className="mt-3 rounded-md border border-yellow-400/20 bg-yellow-400/10 px-3 py-2 text-xs leading-5 text-yellow-100">
                      Limited data detected. We can diagnose bottlenecks, but recommendations may be less precise. Upload run config for more accurate suggestions.
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="overflow-hidden rounded-lg border border-white/10 bg-black/25">
              <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.03] px-4 py-2">
                <span className="text-xs font-medium text-slate-300">Input payload</span>
                <span className="text-[0.65rem] uppercase tracking-[0.16em] text-slate-600">required</span>
              </div>
              <textarea
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
                placeholder='Paste the contents of summary.json, profiler_trace.json, or a raw event array — e.g. { "events": [...] }'
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
                Accepts <code className="text-slate-400">summary.json</code>, <code className="text-slate-400">profiler_trace.json</code>, raw event arrays, and workflow run summaries.
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
                onClick={() => { setSummary(null); setRaw(""); setError(null); setBundleStatus(null); }}
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
