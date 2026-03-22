import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "Belek Analytics — Churn Intelligence Dashboard",
  description: "AI-powered customer retention analytics by Team Belek",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} min-h-screen bg-surface-950 flex noise-bg antialiased`}>
        {/* Animated mesh background */}
        <div className="mesh-gradient">
          <div className="blob blob-1" />
          <div className="blob blob-2" />
          <div className="blob blob-3" />
        </div>

        {/* Global Persistent Sidebar */}
        <Sidebar />

        {/* Dynamic Route Content */}
        <section className="flex-1 overflow-x-hidden overflow-y-auto relative z-10">
          <div className="max-w-[1440px] mx-auto">
            {children}
          </div>
        </section>
      </body>
    </html>
  );
}
