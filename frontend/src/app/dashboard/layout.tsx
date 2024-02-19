"use client";
import Link from "next/link";
import { type ReactNode, useState } from "react";

const sidebarLinks = [
  {
    Name: "My info",
    Link: "/dashboard/profile",
  },
  {
    Name: "Usage",
    Link: "/dashboard/usage",
  },
  {
    Name: "Settings",
    Link: "/dashboard/settings",
  },
];

export default function DashBoardLayout({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <main className="flex-grow">
        {/* Toggle button on left for mobile */}
        <button
          className={`fixed left-4 top-4 z-50 md:hidden ${
            isOpen ? "opacity-100" : "opacity-0"
          }`}
          onClick={() => setIsOpen(!isOpen)}
        >
          Toggle Sidebar
        </button>
        <aside
          className={`w-full bg-gray-200 px-4 py-6 md:w-64 ${
            isOpen ? "block" : "hidden"
          }`}
        >
          <ul className="space-y-4">
            {sidebarLinks.map((link) => (
              <li
                key={link.Name}
                className="text-gray-600 hover:bg-gray-300 hover:text-gray-800"
              >
                <Link
                  href={link.Link}
                  className="block px-4 py-2 font-semibold"
                >
                  {link.Name}
                </Link>
              </li>
            ))}
          </ul>
        </aside>
        <div className="overflow-auto bg-white p-4">{children}</div>
      </main>
      {/* Hide button on larger screens */}
      <button
        className="right-4 top-4 z-50 hidden md:fixed"
        onClick={() => setIsOpen(!isOpen)}
      >
        Toggle Sidebar
      </button>
    </div>
  );
}
