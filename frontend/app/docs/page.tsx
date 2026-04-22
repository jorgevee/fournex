import type { Metadata } from "next";
import Link from "next/link";
import ConstructionSVG from "./construct_svg";

export const metadata: Metadata = {
  title: "Docs - Fournex",
  description: "Fournex documentation is under construction.",
};

export default async function DocsPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(109,40,217,0.22),_transparent_22%),radial-gradient(circle_at_82%_18%,_rgba(56,189,248,0.12),_transparent_18%),linear-gradient(180deg,_#03040b_0%,_#070815_38%,_#04050d_100%)] text-white">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:120px_120px] [mask-image:radial-gradient(circle_at_top,black,transparent_82%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top,_rgba(124,58,237,0.22),_transparent_58%)] blur-3xl" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-[92rem] items-center px-6 py-16 sm:px-8 lg:px-10 xl:px-12">
        <div className="grid w-full items-center gap-12 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="max-w-2xl">
            <div className="inline-flex items-center rounded-full border border-amber-400/25 bg-amber-400/10 px-4 py-2 text-[0.78rem] font-semibold text-amber-200">
              Under construction
            </div>
            <h1 className="mt-6 text-5xl font-semibold tracking-[-0.07em] text-white sm:text-6xl lg:text-[5rem] lg:leading-[0.95]">
              Docs are being
              <br />
              built right now.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300">
              The documentation hub is not ready yet. We are still writing the
              guides, API references, and workflow docs for the product.
            </p>
            <p className="mt-4 max-w-xl text-sm leading-7 text-slate-400">
              In the meantime, you can explore the analyzer, read the product
              walkthrough, or come back once the first docs set ships.
            </p>

            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-violet-600 to-fuchsia-500 px-6 py-4 text-sm font-semibold text-white transition hover:from-violet-500 hover:to-fuchsia-400"
              >
                See how it works
              </Link>
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center rounded-2xl border border-white/12 bg-white/[0.03] px-6 py-4 text-sm font-semibold text-white transition hover:border-white/25 hover:bg-white/[0.06]"
              >
                Open analyzer
              </Link>
            </div>
          </div>

          <div className="flex justify-center lg:justify-end">
            <div className="rounded-[2rem] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] p-6 shadow-[0_35px_100px_rgba(3,6,18,0.45)] sm:p-8">
              <ConstructionSVG className="h-auto w-full max-w-[24rem] sm:max-w-[28rem]" />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
