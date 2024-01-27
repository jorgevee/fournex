"use client";
import { useState } from "react";
import { api } from "~/trpc/react";
import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";

export function SignInForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const logInUser = api.user.loginUser.useMutation({
    onSuccess: () => {
      try {
        await signIn("credentials", {
          email: email,
          password: password,
          redirect: false,
        });
        // Handle successful sign-in
      } catch (error) {
        // Handle errors
      }
      router.push("/dashboard");
    },
    onError: (err) => {
      setError(err.message);
    },
  });

  return (
    <form
      className="mt-6"
      onSubmit={(e) => {
        e.preventDefault();
        logInUser.mutate({ email, password });
      }}
    >
      <div>
        <label htmlFor="email" className="block text-gray-700">
          Email
        </label>

        <input
          type="email"
          id="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mt-2 block w-full rounded-lg border bg-gray-200 px-4 py-3 focus:border-blue-500 focus:bg-white focus:outline-none"
        />
      </div>

      <div className="mt-4">
        <label htmlFor="password" className="block text-gray-700">
          Password
        </label>

        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-2 block w-full rounded-lg border bg-gray-200 px-4 py-3 focus:border-blue-500 focus:bg-white focus:outline-none"
        />
      </div>

      <button
        type="submit"
        className="mt-6 w-full rounded-lg bg-blue-600 px-6 py-3 text-white hover:bg-blue-900"
      >
        Log In
      </button>
    </form>
  );
}
