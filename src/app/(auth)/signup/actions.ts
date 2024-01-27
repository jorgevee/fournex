import { signIn } from "next-auth/react";

export const signInAuth = async (email, password) => {
  await signIn("credentials", {
    email,
    password,
  });
};
