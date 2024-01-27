import type { Metadata } from "next";
import { CreateUser } from "./userForm";
export const metadata: Metadata = {
  title: "Sign Up - Fournex",
  description:
    "Securely log in to your account on My App. Enter your email address and password to get started.",
};

//using drizzle create a post request to create a user.

export default async function SignUp() {
  return (
    <div className="container m-2 mx-auto max-w-md ">
      <h2>Sign Up to Fournex</h2>
      <CreateUser />
    </div>
  );
}
