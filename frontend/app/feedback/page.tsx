"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Loader2,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";

type Outcome = "accepted" | "rejected" | string;

interface FeedbackRecord {
  run_id: string;
  rec_id: string;
  outcome: Outcome;
  created_at?: string | null;
  timestamp?: string | null;
}

type UnknownRecord = Record<string, unknown>;

function asString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

function normalizeRecord(value: unknown, index: number): FeedbackRecord | null {
  if (!value || typeof value !== "object") return null;

  const item = value as UnknownRecord;
  const runId =
    asString(item.run_id) ??
    asString(item.runId) ??
    asString(item.run) ??
    `unknown-run-${index + 1}`;
  const recId =
    asString(item.rec_id) ??
    asString(item.recId) ??
    asString(item.recommendation_id) ??
    asString(item.recommendationId) ??
    asString(item.id);
  const outcome =
    asString(item.outcome) ??
    asString(item.feedback) ??
    asString(item.status) ??
    "unknown";

  if (!recId) return null;

  return {
    run_id: runId,
    rec_id: recId,
    outcome,
    created_at: asString(item.created_at) ?? asString(item.createdAt),
    timestamp: asString(item.timestamp),
  };
}

function normalizePayload(payload: unknown): FeedbackRecord[] {
  const list =
    Array.isArray(payload)
      ? payload
      : payload && typeof payload === "object"
        ? (payload as Record<string, unknown>).feedback ??
          (payload as Record<string, unknown>).items ??
          (payload as Record<string, unknown>).records ??
          (payload as Record<string, unknown>).data ??
          []
        : [];

  if (!Array.isArray(list)) return [];

  return list
    .map((item, index) => normalizeRecord(item, index))
    .filter((item): item is FeedbackRecord => item !== null);
}

function prettyOutcome(outcome: string) {
  if (outcome === "accepted") return "Accepted";
  if (outcome === "rejected") return "Rejected";
  return outcome.replace(/_/g, " ");
}

function formatTime(value?: string | null) {
  if (!value) return "No timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export default function FeedbackPage() {
  const [records, setRecords] = useState<FeedbackRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch("/api/feedback", { cache: "no-store" });
        if (!res.ok) {
          throw new Error(`Feedback API returned ${res.status}`);
        }
        const data = await res.json();
        if (!active) return;
        setRecords(normalizePayload(data));
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load feedback");
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  const summary = useMemo(() => {
    const accepted = records.filter((item) => item.outcome === "accepted").length;
    const rejected = records.filter((item) => item.outcome === "rejected").length;
    const runs = new Set(records.map((item) => item.run_id)).size;
    const uniqueRecs = new Set(records.map((item) => item.rec_id)).size;

    const recommendationMap = new Map<
      string,
      { rec_id: string; accepted: number; rejected: number; total: number }
    >();
    const runMap = new Map<
      string,
      { run_id: string; accepted: number; rejected: number; items: FeedbackRecord[] }
    >();

    for (const record of records) {
      const recEntry = recommendationMap.get(record.rec_id) ?? {
        rec_id: record.rec_id,
        accepted: 0,
        rejected: 0,
        total: 0,
      };
      recEntry.total += 1;
      if (record.outcome === "accepted") recEntry.accepted += 1;
      if (record.outcome === "rejected") recEntry.rejected += 1;
      recommendationMap.set(record.rec_id, recEntry);

      const runEntry = runMap.get(record.run_id) ?? {
        run_id: record.run_id,
        accepted: 0,
        rejected: 0,
        items: [],
      };
      if (record.outcome === "accepted") runEntry.accepted += 1;
      if (record.outcome === "rejected") runEntry.rejected += 1;
      runEntry.items.push(record);
      runMap.set(record.run_id, runEntry);
    }

    return {
      accepted,
      rejected,
      runs,
      uniqueRecs,
      topRecommendations: Array.from(recommendationMap.values())
        .sort((a, b) => b.total - a.total || b.accepted - a.accepted)
        .slice(0, 8),
      runSummaries: Array.from(runMap.values()).sort(
        (a, b) => b.items.length - a.items.length,
      ),
    };
  }, [records]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(109,40,217,0.22),_transparent_22%),radial-gradient(circle_at_85%_20%,_rgba(56,189,248,0.12),_transparent_18%),linear-gradient(180deg,_#03040b_0%,_#070815_38%,_#04050d_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:120px_120px] [mask-image:radial-gradient(circle_at_top,black,transparent_82%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top,_rgba(124,58,237,0.22),_transparent_58%)] blur-3xl" />

      <div className="relative mx-auto flex w-full max-w-[92rem] flex-col px-6 pb-24 pt-10 sm:px-8 lg:px-10 xl:px-12">
        <section className="grid gap-10 border-b border-white/8 pb-12 pt-10 lg:grid-cols-[0.95fr_1.05fr] lg:items-end">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-2 text-[0.78rem] font-semibold text-violet-200">
              <CheckCircle2 size={14} className="text-violet-300" />
              Feedback loop dashboard
            </div>
            <h1 className="mt-6 text-5xl font-semibold tracking-[-0.07em] text-white sm:text-6xl lg:text-[5rem] lg:leading-[0.95]">
              Past recommendation
              <br />
              outcomes across
              <span className="text-violet-400"> runs.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300">
              This view closes the loop on recommendations. See which fixes users
              accepted or rejected, which recommendations show up most often,
              and how feedback is distributed across prior runs.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { label: "Accepted", value: summary.accepted.toString() },
              { label: "Rejected", value: summary.rejected.toString() },
              { label: "Runs tracked", value: summary.runs.toString() },
              { label: "Unique recs", value: summary.uniqueRecs.toString() },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-[1.4rem] border border-white/8 bg-white/[0.03] p-5"
              >
                <div className="text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-slate-500">
                  {item.label}
                </div>
                <div className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="py-10">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm text-slate-400">
              Using stored feedback submitted from the thumbs up/down controls in
              the analyzer.
            </div>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-2 rounded-xl border border-white/12 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white transition hover:border-white/25 hover:bg-white/[0.06]"
            >
              <RefreshCw size={14} />
              Refresh
            </button>
          </div>

          {loading ? (
            <div className="flex min-h-[240px] items-center justify-center rounded-[1.8rem] border border-white/8 bg-white/[0.03]">
              <div className="inline-flex items-center gap-3 text-slate-300">
                <Loader2 size={18} className="animate-spin" />
                Loading feedback history
              </div>
            </div>
          ) : error ? (
            <div className="rounded-[1.8rem] border border-red-400/20 bg-red-400/10 p-6 text-sm text-red-100">
              Failed to load feedback data: {error}
            </div>
          ) : records.length === 0 ? (
            <div className="rounded-[1.8rem] border border-white/8 bg-white/[0.03] p-8">
              <h2 className="text-xl font-semibold text-white">No feedback yet</h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-400">
                Submit thumbs up/down feedback from the analyzer page and the
                results will appear here once the backend returns stored records.
              </p>
              <Link
                href="/analyze"
                className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-5 py-3 text-sm font-semibold text-white transition hover:from-violet-500 hover:to-fuchsia-400"
              >
                Go to analyzer
                <ArrowRight size={14} />
              </Link>
            </div>
          ) : (
            <div className="grid gap-6 xl:grid-cols-[0.88fr_1.12fr]">
              <div className="space-y-6">
                <div className="rounded-[1.8rem] border border-white/8 bg-white/[0.03] p-6">
                  <h2 className="text-lg font-semibold text-white">
                    Most-seen recommendations
                  </h2>
                  <div className="mt-5 space-y-3">
                    {summary.topRecommendations.map((item) => (
                      <div
                        key={item.rec_id}
                        className="rounded-[1.2rem] border border-white/8 bg-[#090c18] px-4 py-4"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-medium text-white">
                              {item.rec_id}
                            </div>
                            <div className="mt-1 text-xs text-slate-500">
                              {item.total} total responses
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-3 text-xs">
                            <span className="inline-flex items-center gap-1 text-emerald-300">
                              <ThumbsUp size={12} />
                              {item.accepted}
                            </span>
                            <span className="inline-flex items-center gap-1 text-red-300">
                              <ThumbsDown size={12} />
                              {item.rejected}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-[1.8rem] border border-white/8 bg-white/[0.03] p-6">
                  <h2 className="text-lg font-semibold text-white">
                    Raw feedback stream
                  </h2>
                  <div className="mt-5 space-y-3">
                    {records.slice(0, 12).map((record, index) => (
                      <div
                        key={`${record.run_id}-${record.rec_id}-${index}`}
                        className="rounded-[1.2rem] border border-white/8 bg-[#090c18] px-4 py-4"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-4">
                          <div>
                            <div className="text-sm font-medium text-white">
                              {record.rec_id}
                            </div>
                            <div className="mt-1 text-xs text-slate-500">
                              Run {record.run_id}
                            </div>
                          </div>
                          <div
                            className={`rounded-full px-3 py-1 text-xs font-medium ${
                              record.outcome === "accepted"
                                ? "bg-emerald-400/10 text-emerald-300"
                                : record.outcome === "rejected"
                                  ? "bg-red-400/10 text-red-300"
                                  : "bg-white/5 text-slate-300"
                            }`}
                          >
                            {prettyOutcome(record.outcome)}
                          </div>
                        </div>
                        <div className="mt-3 text-xs text-slate-500">
                          {formatTime(record.created_at ?? record.timestamp)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-[1.8rem] border border-white/8 bg-white/[0.03] p-6">
                <h2 className="text-lg font-semibold text-white">
                  Outcomes by run
                </h2>
                <div className="mt-5 space-y-4">
                  {summary.runSummaries.map((run) => (
                    <article
                      key={run.run_id}
                      className="rounded-[1.3rem] border border-white/8 bg-[#090c18] p-5"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <div className="text-sm font-semibold text-white">
                            {run.run_id}
                          </div>
                          <div className="mt-1 text-xs text-slate-500">
                            {run.items.length} feedback event
                            {run.items.length !== 1 ? "s" : ""}
                          </div>
                        </div>
                        <div className="flex items-center gap-4 text-xs">
                          <span className="inline-flex items-center gap-1 text-emerald-300">
                            <ThumbsUp size={12} />
                            {run.accepted}
                          </span>
                          <span className="inline-flex items-center gap-1 text-red-300">
                            <ThumbsDown size={12} />
                            {run.rejected}
                          </span>
                        </div>
                      </div>

                      <div className="mt-4 overflow-hidden rounded-2xl border border-white/8">
                        <table className="w-full text-left text-sm">
                          <thead className="bg-white/[0.03] text-xs uppercase tracking-[0.18em] text-slate-500">
                            <tr>
                              <th className="px-4 py-3 font-medium">Recommendation</th>
                              <th className="px-4 py-3 font-medium">Outcome</th>
                              <th className="px-4 py-3 font-medium">Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {run.items.map((record, index) => (
                              <tr
                                key={`${record.rec_id}-${index}`}
                                className="border-t border-white/8"
                              >
                                <td className="px-4 py-3 text-slate-200">
                                  {record.rec_id}
                                </td>
                                <td className="px-4 py-3">
                                  <span
                                    className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                                      record.outcome === "accepted"
                                        ? "bg-emerald-400/10 text-emerald-300"
                                        : record.outcome === "rejected"
                                          ? "bg-red-400/10 text-red-300"
                                          : "bg-white/5 text-slate-300"
                                    }`}
                                  >
                                    {prettyOutcome(record.outcome)}
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-xs text-slate-500">
                                  {formatTime(record.created_at ?? record.timestamp)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
