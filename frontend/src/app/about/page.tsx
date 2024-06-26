/**
 * v0 by Vercel.
 * @see https://v0.dev/t/VbvyYPtomLc
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
import Image from "next/image";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About Us - Fournex",
  description:
    "Learn more about Fournex, an AI startup focused on developing Autonomous Agents.",
};

export default async function Component() {
  return (
    <section className="w-full py-6 md:py-12 lg:py-24 xl:py-32">
      <div className="container px-4 md:px-6">
        <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-2 xl:grid-cols-2">
          <Image
            alt="About Us"
            className="mx-auto aspect-video w-full overflow-hidden rounded-xl object-cover object-center md:w-3/4 lg:order-last lg:w-full"
            src="/ai.png"
            width={350}
            height={350}
          />
          <div className="flex flex-col justify-center space-y-4">
            <div className="space-y-2">
              <h1 className="text-2xl font-bold tracking-tighter md:text-3xl lg:text-4xl xl:text-5xl">
                About Us
              </h1>
              <h2 className="text-xl font-bold tracking-tighter md:text-3xl lg:text-4xl xl:text-5xl">
                We are 100% open sourced
              </h2>
              <p className="text-sm text-gray-500 md:text-base lg:text-lg xl:text-xl dark:text-gray-400">
                Fournex is an AI startup focused on developing techniques for
                developing Autonomous Agents. We utilize DSPy, a framework for
                algorithmically optimizing language model prompts and weights.
                DSPy allows us to treat language models as optimizable
                components within a larger machine learning system.
              </p>
              <p className="text-sm text-gray-500 md:text-base lg:text-lg xl:text-xl dark:text-gray-400">
                By separating the system logic from the model parameters, We
                enable the use of novel &quot;optimizers&quot; - language
                model-driven algorithms that can tune prompts and weights to
                maximize a given metric. Rather than manually prompting and
                finetuning models through trial and error, we automate this
                process. Our optimizers can routinely teach models new skills
                and avoid failure modes through data-driven parameter
                optimization. This represents a paradigm shift where language
                models fade into the background as optimized pieces of a system
                that can continuously learn from experience.
              </p>
              <p className="text-sm text-gray-500 md:text-base lg:text-lg xl:text-xl dark:text-gray-400">
                At Fournex, our primary objective is to pioneer the development
                of powerful and productive AI Autonomous agents. We firmly
                believe that the ongoing research and advancement as well as
                frameworks such as DSPy, LangGraph (LangChain) and LLamaIndex
                model optimization are pivotal in creating robust and secure
                language model applications. Our focus is on devising techniques
                that enhance the systematic reliability of models, thereby
                minimizing the necessity for extensive human oversight.
              </p>
            </div>
            <div className="flex flex-col gap-2 md:flex-row">
<Link 
  className="
    inline-flex items-center justify-center 
    rounded-lg bg-blue-500 
    p-2 text-sm font-medium text-white 
    transition duration-300 ease-in-out 
    hover:bg-blue-600 focus:bg-blue-700 
    focus:outline-none focus:ring-2 focus:ring-blue-300
  " 
  href="/"
>
  <span className="mr-2">Back To Home</span>
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
</Link>

    
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
