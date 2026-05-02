import type { Metadata } from "next";
import WaitlistSignup from "./coming-soon-comp";

export const metadata: Metadata = {
  title: "Join the Fournex waitlist",
  description: "Join the Fournex early access waitlist and Discord community.",
};

export default function LoginPage() {
  return (
    <main>
      <WaitlistSignup />
    </main>
  );
}
