"use client";
import Link from "next/link";
import { useState } from "react";
import { useSession } from "next-auth/react";
import { HiOutlineX, HiMenuAlt3 } from "react-icons/hi";
import { IoMdClose } from "react-icons/io";
const links = [
  { text: "About", href: "/about" },
  { text: "Contact", href: "/contact" },
];

export default function Header() {
  const { data: session, status } = useSession();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleMobileMenuToggle = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  return (
    <header className="bg-slate-900 p-4 text-white">
      <div className="container mx-auto">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-2xl font-bold">
            Fournex
          </Link>
          <button
            className="block focus:outline-none lg:hidden"
            onClick={handleMobileMenuToggle}
          >
            {mobileMenuOpen ? (
              <IoMdClose size={24} />
            ) : (
              <HiMenuAlt3 size={24} />
            )}
          </button>
          <nav className="hidden w-full lg:block">
            <ul
              className={`justify-end transition-all duration-300 lg:flex lg:items-center lg:space-x-4 ${
                mobileMenuOpen ? "block" : "hidden"
              }`}
            >
              {links.map((link, index) => (
                <li key={index} className="text-center">
                  <Link
                    href={link.href}
                    className="theme-link block py-2 lg:inline-block lg:px-4 lg:py-0"
                  >
                    {link.text}
                  </Link>
                </li>
              ))}
              <div className="lg:flex lg:items-center lg:justify-end lg:space-x-4">
                {session ? (
                  <li className="text-center">
                    <Link
                      className="rounded-lg bg-blue-500 px-4 py-2 font-semibold text-white hover:bg-blue-700"
                      href="/api/auth/signout"
                    >
                      Sign Out
                    </Link>
                  </li>
                ) : (
                  <>
                    <li className="text-center">
                      <Link
                        className="mr-2 rounded-lg bg-white px-4 py-2 font-semibold text-black hover:bg-gray-200"
                        href="/login"
                      >
                        Log In
                      </Link>
                    </li>
                    <li className="text-center">
                      <Link
                        className="rounded-lg bg-blue-700 px-4 py-2 font-semibold text-white hover:bg-blue-500"
                        href="/signup"
                      >
                        Sign Up
                      </Link>
                    </li>
                  </>
                )}
              </div>
            </ul>
          </nav>
        </div>

        <nav className="lg:hidden">
          {mobileMenuOpen && (
            <div className="animate-open-menu absolute left-0 top-16 z-10 w-full origin-top">
              <div className="rounded-lg bg-slate-800 p-4">
                <button
                  className="float-right"
                  onClick={handleMobileMenuToggle}
                ></button>

                <ul className="space-y-2 text-center">
                  {links.map((link, index) => (
                    <li key={index}>
                      <Link
                        href={link.href}
                        className="block py-2 text-white hover:text-gray-300"
                      >
                        {link.text}
                      </Link>
                    </li>
                  ))}
                </ul>

                {session ? (
                  <div className="mt-4 text-center">
                    <Link
                      className="block rounded-lg bg-blue-500 px-4 py-2 font-semibold text-white hover:bg-blue-700"
                      href="/api/auth/signout"
                    >
                      Sign Out
                    </Link>
                  </div>
                ) : (
                  <div className="mt-4 space-y-2 text-center">
                    <Link
                      className="block rounded-lg bg-white px-4 py-2 font-semibold text-black hover:bg-gray-200"
                      href="/login"
                    >
                      Log In
                    </Link>
                    <Link
                      className="block rounded-lg bg-blue-700 px-4 py-2 font-semibold text-white hover:bg-blue-500"
                      href="/signup"
                    >
                      Sign Up
                    </Link>
                  </div>
                )}
              </div>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
}
