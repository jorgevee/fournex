import type { Metadata } from "next";
import LoginForm from "./loginForm";

export const metadata: Metadata = {
  title: "Login - Fournex",
  description:
    "Securely log in to your account on My App. Enter your email address and password to get started.",
};


export default async function LoginPage() {
  return (
     <div className="flex flex-col min-h-screen justify-center items-center px-6 py-12 lg:px-8 bg-gray-50">
       <h2 className="mb-4 text-2xl font-bold">Welcome Back</h2>
       <p className="mb-8 max-w-xl text-lg text-gray-500">
         Fournex enables building stateful, multi-agent applications with LLMs. It
         coordinates multiple AI agents to interact in a cyclic workflow.
       </p>
       <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-md">
         <LoginForm />
       </div>
     </div>
  );
 }
 
