import "~/styles/globals.css";

import { Inter } from "next/font/google";
import { GoogleAnalytics } from "@next/third-parties/google";
import Header from "./_components/layout/header";
import Footer from "./_components/layout/footer";
import NextAuthProvider from "./_context/ClientAuth";
import { TRPCReactProvider } from "~/trpc/react";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});


export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`font-sans ${inter.variable}`}>
          <TRPCReactProvider>
            <Header />
            {children}
            <Footer />
          </TRPCReactProvider>
      </body>
      <GoogleAnalytics gaId="G-00T268ELW1" />
    </html>
  );
}
