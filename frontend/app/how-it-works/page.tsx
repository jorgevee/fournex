import type { Metadata } from "next";
import Hero from "../_components/heromain";

export const metadata: Metadata = {
  title: "How It Works",
  description: "Learn how our semiconductor EDA tools work."
};

export default function HowItWorksPage() {
  return (
    <div>
      <Hero />
    </div>
  );
}