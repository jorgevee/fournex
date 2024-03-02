"use client";
import Link from "next/link";
import { type ReactNode, useState } from "react";

const sidebarLinks = [
  {
    Name: "My info",
    Link: "/dashboard",
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
    <div className="flex flex-col h-screen">
      <div className="flex-1 flex flex-col">
        <div className="flex-1 flex flex-col">
          {/* Sidebar */}
          <div className="flex flex-col w-64 bg-gray-100 border-r border-gray-200 p-4">
            <div className="flex flex-col items-center justify-between">
              <span className="text-sm text-gray-500">Dashboard</span>
              <button
                type="button"
                className="ml-auto flex items-center justify-center w-10 h-10 rounded-full bg-gray-200 text-gray-500 hover:bg-gray-300"
                onClick={() => setIsOpen(!isOpen)}
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            </div>
            <div className="mt-4 flex flex-col">
              {sidebarLinks.map((link) => (
                <Link
                  key={link.Name}
                  href={link.Link}
                  className="flex items-center justify-between w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900"
                >
                  <span>{link.Name}</span>
                </Link>
              ))}
            </div>
          </div>
          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-y-auto">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
