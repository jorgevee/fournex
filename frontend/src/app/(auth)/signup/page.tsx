import type { Metadata } from "next";
import SignUpForm from "./signUpForm";
export const metadata: Metadata = {
  title: "Sign Up - Fournex",
  description:
    "Securely sign up for your account on Fournex. Enter your email address and password to get started.",
};

export default async function SignUpPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-gray-900">
            Sign Up to Fournex
          </h2>
          <p className="mt-2 text-lg text-gray-600">
            Join our community to enjoy the benefits of Fournex.
          </p>
        </div>

        <SignUpForm />
        {/* Optional additional elements */}
        <p className="text-center text-gray-600 text-sm">
          Already have an account? <a href="/login" className="font-medium text-blue-600 hover:text-blue-500">
            Log in
          </a>
        </p>
      </div>
    </div>
  );
}
