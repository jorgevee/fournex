"use client";
import { type ChangeEvent, type FormEvent, useState } from "react";
import { signIn } from "next-auth/react";
import { api } from "~/trpc/react";
type LoginInput = {
  email: string;
  password: string;
};

export default function LoginForm() {
    const [inputs, setInputs] = useState<LoginInput>({
        email: "",
        password: "",
    });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const login = api.user.loginUser.useMutation({
        onSuccess: async () => {
            setLoading(true);
          await signIn("credentials", {
            email: inputs.email,
            password: inputs.password,
            callbackUrl:"/dashboard",
            redirect: true,
          });
          setLoading(false);
        },
        onError: () => {
          // If error display erorr message
          setError("Invalid email or password");
        },
      });

    async function handleSubmit(e: FormEvent) {
        e.preventDefault();
        login.mutate({ email: inputs.email, password: inputs.password });
    }

  return (
    <form onSubmit={handleSubmit}>
      <div className="mb-4">
        <label htmlFor="email" className="block font-bold text-gray-700">
          Email
        </label>
        <input
          type="email"
          id="email"
          className="w-full rounded border border-gray-400 p-2"
          value={inputs.email || ""}
          onChange={(e) => setInputs({ ...inputs, email: e.target.value })}
          required={true}
          />
      </div>

      <div className="mb-4">
        <label htmlFor="message" className="block font-bold text-gray-700">
          Password
        </label>
        <input
          type="password"
          id="password"
          className="w-full rounded border border-gray-400 p-2"
          value={inputs.password || ""}
          onChange={(e) => setInputs({ ...inputs, password: e.target.value })}
          required={true}
        />
      </div>

      <div className="mb-4">
        <button
          type="submit"
          className="flex w-full justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          disabled={loading}
        >
          Submit
        </button>
      </div>
      {loading ? (
        <p className="text-center capitalize text-indigo-600">
          Signing in...
        </p>
      ) : null}
      {error && <div className="text-red-500">{error}</div>}
    </form>
  );
}