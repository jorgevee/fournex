"use client";

import Link from "next/link";
import { ArrowRight, CheckCircle2, Loader2, Mail, MessageCircle, ShieldCheck } from "lucide-react";
import { FormEvent, useState } from "react";

type SubmitState = "idle" | "submitting" | "success" | "error";

const discordUrl = process.env.NEXT_PUBLIC_DISCORD_URL ?? "https://discord.com";

export default function WaitlistSignup() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<SubmitState>("idle");
  const [message, setMessage] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedEmail) {
      setStatus("error");
      setMessage("Enter an email address to join the waitlist.");
      return;
    }

    setStatus("submitting");
    setMessage("");

    try {
      const response = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: normalizedEmail, source: "login_waitlist" }),
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.error ?? "Could not join the waitlist.");
      }

      setStatus("success");
      setMessage("You are on the waitlist. We will send early access details when your spot opens.");
      setEmail("");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not join the waitlist.");
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.18),_transparent_26%),radial-gradient(circle_at_85%_18%,_rgba(124,58,237,0.18),_transparent_24%),linear-gradient(180deg,_#03040b_0%,_#070815_44%,_#04050d_100%)] px-4 py-10 text-white sm:px-6 lg:px-8">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.05)_1px,transparent_1px)] bg-[size:96px_96px] [mask-image:radial-gradient(circle_at_top,black,transparent_78%)]" />
      <div className="relative mx-auto flex min-h-[calc(100vh-5rem)] max-w-5xl items-center">
        <div className="grid w-full gap-8 lg:grid-cols-[0.95fr_1.05fr] lg:items-center">
          <section className="max-w-2xl">
            <Link href="/" className="inline-flex items-center gap-2 text-sm text-slate-300 transition hover:text-white">
              <ArrowRight size={14} className="rotate-180" />
              Back to Fournex
            </Link>

            <div className="mt-10 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200">
              <ShieldCheck size={13} />
              Early access
            </div>

            <h1 className="mt-5 max-w-2xl text-4xl font-semibold tracking-[-0.05em] text-white sm:text-5xl lg:text-6xl lg:leading-[1.02]">
              Join Fournex waitlist.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-slate-300 sm:text-lg sm:leading-8">
              Get access to Fournex as we open hosted accounts for teams tuning PyTorch and NVIDIA workloads.
            </p>

            <div className="mt-8 grid gap-3 text-sm text-slate-300 sm:grid-cols-3">
              {["Safe tuning loops", "CLI-first workflow", "Discord support"].map((item) => (
                <div key={item} className="flex items-center gap-2 rounded-xl border border-white/8 bg-white/[0.04] px-3 py-2">
                  <CheckCircle2 size={15} className="shrink-0 text-cyan-300" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-[1.5rem] border border-white/10 bg-slate-950/80 p-5 shadow-[0_30px_100px_rgba(2,6,23,0.55)] backdrop-blur-xl sm:rounded-[1.75rem] sm:p-6 lg:p-8">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 text-cyan-200">
              <Mail size={22} />
            </div>
            <h2 className="mt-5 text-2xl font-semibold tracking-[-0.03em] text-white">
              Request early access
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              We will use this email only for waitlist updates and onboarding.
            </p>

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              <label className="block text-sm font-medium text-slate-300">
                Work email
                <input
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  type="email"
                  name="email"
                  autoComplete="email"
                  placeholder="team@company.com"
                  className="mt-2 w-full rounded-xl border border-white/10 bg-white/[0.05] px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50 focus:bg-white/[0.08]"
                  disabled={status === "submitting"}
                  required={true}
                />
              </label>

              {message && (
                <div
                  className={`rounded-xl border px-4 py-3 text-sm ${
                    status === "success"
                      ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-100"
                      : "border-rose-400/30 bg-rose-400/10 text-rose-100"
                  }`}
                >
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={status === "submitting"}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {status === "submitting" ? <Loader2 size={16} className="animate-spin" /> : <Mail size={16} />}
                Join waitlist
              </button>
            </form>

            <div className="mt-5 flex items-center gap-3">
              <div className="h-px flex-1 bg-white/10" />
              <span className="text-xs uppercase tracking-[0.22em] text-slate-500">or</span>
              <div className="h-px flex-1 bg-white/10" />
            </div>

            <a
              href={discordUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.05] px-5 py-3 text-sm font-semibold text-white transition hover:border-white/20 hover:bg-white/[0.08]"
            >
              <MessageCircle size={17} />
              Join Discord
            </a>
          </section>
        </div>
      </div>
    </div>
  );
}
