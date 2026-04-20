import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Fournex — Open-source GPU Optimizer",
  description:
    "Profile your PyTorch training and inference jobs, get the bottleneck named for you, and ship the highest-ROI fix — validated by safe experiments, not hope.",
  openGraph: {
    title: "Fournex — Open-source GPU Optimizer",
    description:
      "Profile your PyTorch training and inference jobs, get the bottleneck named for you, and ship the highest-ROI fix.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
