"use client";

import { useSession } from "next-auth/react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { data: session } = useSession();
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <div className="min-h-screen bg-white">
      <header className="fixed inset-x-0 top-0 z-50 bg-white/80 backdrop-blur-sm">
        <nav
          className="flex items-center justify-between p-6 lg:px-8"
          aria-label="Global"
        >
          <div className="flex lg:flex-1">
            <Link href="/" className="-m-1.5 p-1.5">
              <span className="text-xl font-bold text-blue-600">SwiftAid</span>
            </Link>
          </div>
          <div className="flex lg:hidden">
            <button
              type="button"
              className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
            >
              <span className="sr-only">Open main menu</span>
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth="1.5"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
                />
              </svg>
            </button>
          </div>
          <div className="hidden lg:flex lg:gap-x-12">
            <Link
              href="/explore"
              className={`text-sm font-semibold leading-6 ${
                isActive("/explore")
                  ? "text-blue-600"
                  : "text-gray-900 hover:text-blue-600"
              }`}
            >
              Explore
            </Link>
            <Link
              href="/services"
              className={`text-sm font-semibold leading-6 ${
                isActive("/services")
                  ? "text-blue-600"
                  : "text-gray-900 hover:text-blue-600"
              }`}
            >
              Services
            </Link>
            <Link
              href="/preferences"
              className={`text-sm font-semibold leading-6 ${
                isActive("/preferences")
                  ? "text-blue-600"
                  : "text-gray-900 hover:text-blue-600"
              }`}
            >
              Preferences
            </Link>
          </div>
          <div className="hidden lg:flex lg:flex-1 lg:justify-end">
            {session ? (
              <Link
                href="/profile"
                className="text-sm font-semibold leading-6 text-gray-900 hover:text-blue-600"
              >
                Profile
              </Link>
            ) : (
              <Link
                href="/auth/login"
                className="text-sm font-semibold leading-6 text-gray-900 hover:text-blue-600"
              >
                Log in
              </Link>
            )}
          </div>
        </nav>
      </header>
      <main className="pt-24">{children}</main>
    </div>
  );
}
