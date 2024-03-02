import Image from "next/image";
import type { Metadata } from "next/types";

export const metadata: Metadata = {
  title: "Create AI Agents - Fournex",
  description: "Create AI autonomus agents using Fournex. Build stateful, multi-agent applications with LLMs.",
  
};

export default async function Main() {
  return (
    <main>
      <HeroTop />
      <HeroSection />
    </main>
  );
}

function HeroSection() {
  return (
    <section className="w-full bg-black py-6 md:py-24 lg:py-32 xl:py-48">
      <div className="container mx-auto px-2 md:px-6">
        <div className="grid items-center gap-4 md:gap-6">
          <div className="flex flex-col justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="bg-gradient-to-r from-white to-gray-500 bg-clip-text text-2xl font-bold tracking-tighter text-transparent sm:text-5xl xl:text-6xl">
                Meet Our Intelligent AI Agents
              </h1>
              <p className="mx-auto max-w-[600px] text-zinc-200 md:text-xl">
                Our AI agents are powered by state-of-the-art language models to
                provide helpful, personalized assistance.
              </p>
            </div>
            <div className="mx-auto w-full max-w-full space-y-4">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3 md:gap-8">
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <InboxIcon className="h-6 w-6 text-white opacity-75" />
                  </div>
                  <h2 className="text-xl font-bold text-white">
                    Conversational
                  </h2>
                  <p className="text-zinc-200">
                    Our agents can understand natural language and carry on
                    helpful conversations.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <LockIcon className="h-6 w-6 text-white opacity-75" />
                  </div>
                  <h2 className="text-xl font-bold text-white">Intelligent</h2>
                  <p className="text-zinc-200">
                    With advanced language models, our agents can reason,
                    recommend, and explain.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <MergeIcon className="h-6 w-6 text-white opacity-75" />
                  </div>
                  <h2 className="text-xl font-bold text-white">Secure</h2>
                  <p className="text-zinc-200">
                    We use state-of-the-art techniques to keep your data safe
                    and private.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <SearchIcon className="h-6 w-6 text-white opacity-75" />{" "}
                  </div>
                  <h2 className="text-xl font-bold text-white">Personalized</h2>{" "}
                  <p className="text-zinc-200">
                    Our agents get to know you over time to provide a more
                    tailored experience.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <SearchIcon className="h-6 w-6 text-white opacity-75" />{" "}
                  </div>
                  <h2 className="text-xl font-bold text-white">Customizable</h2>{" "}
                  <p className="text-zinc-200">
                    Customize your agent&apos;s abilities and knowledge based on
                    your needs.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 rounded-lg border-gray-800 p-4">
                  <div className="rounded-full bg-black bg-opacity-50 p-2">
                    <SettingsIcon className="h-6 w-6 text-white opacity-75" />{" "}
                  </div>
                  <h2 className="text-xl font-bold text-white">
                    Task Automation
                  </h2>
                  <p className="text-zinc-200">
                    Use our agents to automate repetitive tasks and workflows.{" "}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function InboxIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
      <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
    </svg>
  );
}

function LockIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function MergeIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m8 6 4-4 4 4" />
      <path d="M12 2v10.3a4 4 0 0 1-1.172 2.872L4 22" />
      <path d="m20 22-5-5" />
    </svg>
  );
}

function SearchIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function SettingsIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function HeroTop() {
  const mainImg = "/assets/mainimg-icon.png";
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
          <Image
            alt="Hero"
            className="mx-auto hidden aspect-video overflow-hidden rounded-xl object-cover object-bottom sm:w-full md:block lg:order-last"
            src={mainImg}
            width={500}
            height={500}
          />
          <div className="flex flex-col justify-center space-y-4">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                Welcome to our AI agent client
              </h1>
              <p className="max-w-[600px] text-zinc-500 md:text-xl dark:text-zinc-400">
                Get on our waitlist today!
              </p>
            </div>
            <div className="w-full max-w-sm space-y-2">
              <form className="flex space-x-2">
                <input
                  className="border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring flex h-10 w-full max-w-lg flex-1 rounded-md border px-3 py-2 text-sm file:border-0 file:bg-transparent file:text-sm file:font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="Enter your email"
                  type="email"
                  required={true}
                />
                <button
                  className="ring-offset-background focus-visible:ring-ring bg-primary text-primary-foreground hover:bg-primary/90 inline-flex h-10 items-center justify-center whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                  type="submit"
                >
                  Start Now
                </button>
              </form>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Start managing your emails today.{" "}
                <a className="underline underline-offset-2" href="#">
                  Terms & Conditions
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
