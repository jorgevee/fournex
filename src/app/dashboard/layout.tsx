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
    <div className="flex min-h-screen flex-col">
      <main className="flex-grow">
        <aside className="w-full flex-shrink-0 bg-gray-200 px-4 py-6 md:w-64">
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
    </div>
  );
}
