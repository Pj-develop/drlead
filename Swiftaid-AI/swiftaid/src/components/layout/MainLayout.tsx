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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="text-xl font-bold text-gray-800">
              SwiftAid
            </Link>
            <div className="flex items-center space-x-4">
              {session ? (
                <>
                  <Link
                    href="/explore"
                    className={`text-gray-600 hover:text-gray-900 ${
                      pathname === "/explore" ? "font-semibold" : ""
                    }`}
                  >
                    Explore
                  </Link>
                  <Link
                    href="/preferences"
                    className={`text-gray-600 hover:text-gray-900 ${
                      pathname === "/preferences" ? "font-semibold" : ""
                    }`}
                  >
                    Preferences
                  </Link>
                </>
              ) : (
                <Link
                  href="/auth/login"
                  className="text-gray-600 hover:text-gray-900"
                >
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">{children}</main>

      {/* Footer */}
      <footer className="bg-white border-t">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-gray-600">
            © {new Date().getFullYear()} SwiftAid. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
