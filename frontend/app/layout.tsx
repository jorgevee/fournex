import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Header from "./_components/layout/header";
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
      <body className="flex min-h-full flex-col">
        <Header />
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </body>
    </html>
  );
}
