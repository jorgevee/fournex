"use client";
import type { Metadata } from "next";
import { type FormEvent, useState } from "react";
import { api } from "~/trpc/react";
import { signIn } from "next-auth/react";
import { TRPCClientError } from "@trpc/client";
// export const metadata: Metadata = {
//   title: "Sign Up - Fournex",
//   description:
//     "Securely log in to your account on My App. Enter your email address and password to get started.",
// };

//using drizzle create a post request to create a user.

export default function SignUp() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const createUser = api.user.createUser.useMutation({
    onSuccess: async () => {
      await signIn("credentials", {
        email: email,
        password: password,
        callbackUrl: "/dashboard",
      });
    },
    onError: (err) => {
      setError(err.message);
    },
  });

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    createUser.mutate({ email, password });
  }
  return (
    <div className="container m-2 mx-auto max-w-md ">
      <h2 className="mb-4 text-2xl font-bold text-gray-900">
        Sign Up to Fournex
      </h2>
      <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
        <div>
          <label
            htmlFor="email"
            className="block text-sm font-medium text-gray-700"
          >
            Email address
          </label>
          <div className="mt-1">
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required={true}
              className="block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 placeholder-gray-400 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm"
            />
          </div>
        </div>

        <div>
          <label
            htmlFor="password"
            className="block text-sm font-medium text-gray-700"
          >
            Password
          </label>
          <div className="mt-1">
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required={true}
              className="block w-full appearance-none rounded-md border border-gray-300 px-3 py-2 placeholder-gray-400 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm"
            />
          </div>
        </div>

        <div>
          <button
            type="submit"
            className="flex w-full justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            Log In
          </button>
        </div>
        {error && <p className="text-red-500">{error}</p>}
      </form>
    </div>
  );
}
