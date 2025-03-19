"use client";

import { SessionProvider } from "next-auth/react";
import MainLayout from "@/components/layout/MainLayout";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <MainLayout>{children}</MainLayout>
    </SessionProvider>
  );
}
