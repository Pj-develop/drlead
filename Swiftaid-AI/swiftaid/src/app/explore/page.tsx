"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { motion } from "framer-motion";
import { trackCityExploration, trackUserAction } from "@/lib/analytics";

export default function ExplorePage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/auth/login");
    }
  }, [status, router]);

  useEffect(() => {
    if (session?.user) {
      trackUserAction("explore_page_visit", {
        userId: session.user.id,
        timestamp: new Date().toISOString(),
      });
    }
  }, [session]);

  const handleCitySelect = (city: string) => {
    trackCityExploration(city, "select", {
      userId: session?.user?.id,
      timestamp: new Date().toISOString(),
    });
    // Add your city selection logic here
  };

  const handleServiceSelect = (service: string) => {
    trackUserAction("service_selection", {
      service,
      userId: session?.user?.id,
      timestamp: new Date().toISOString(),
    });
    // Add your service selection logic here
  };

  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!session) {
    return null;
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.1,
        duration: 0.5,
      },
    }),
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-b from-white to-gray-50">
      {/* Background decoration */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-blue-200 to-blue-400 opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
      </div>

      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-7xl mx-auto"
        >
          <h1 className="text-4xl font-bold tracking-tight text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-blue-800 mb-12">
            Explore Cities
          </h1>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* City Cards */}
            <motion.div
              custom={0}
              initial="hidden"
              animate="visible"
              variants={cardVariants}
              className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-shadow duration-300"
            >
              <h2 className="text-2xl font-semibold mb-6 text-gray-900">
                Popular Cities
              </h2>
              <div className="space-y-3">
                {["New York", "San Francisco", "Chicago", "Los Angeles"].map(
                  (city) => (
                    <button
                      key={city}
                      onClick={() => handleCitySelect(city)}
                      className="w-full text-left p-4 hover:bg-blue-50 rounded-xl transition-all duration-200 text-gray-700 hover:text-blue-600 hover:shadow-md"
                    >
                      {city}
                    </button>
                  )
                )}
              </div>
            </motion.div>

            {/* Services Section */}
            <motion.div
              custom={1}
              initial="hidden"
              animate="visible"
              variants={cardVariants}
              className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-shadow duration-300"
            >
              <h2 className="text-2xl font-semibold mb-6 text-gray-900">
                Available Services
              </h2>
              <div className="space-y-3">
                {["Housing", "Utilities", "Healthcare", "Education"].map(
                  (service) => (
                    <button
                      key={service}
                      onClick={() => handleServiceSelect(service)}
                      className="w-full text-left p-4 hover:bg-blue-50 rounded-xl transition-all duration-200 text-gray-700 hover:text-blue-600 hover:shadow-md"
                    >
                      {service}
                    </button>
                  )
                )}
              </div>
            </motion.div>

            {/* Recent Activity */}
            <motion.div
              custom={2}
              initial="hidden"
              animate="visible"
              variants={cardVariants}
              className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-shadow duration-300"
            >
              <h2 className="text-2xl font-semibold mb-6 text-gray-900">
                Recent Activity
              </h2>
              <div className="space-y-4">
                <p className="text-gray-600 p-4">
                  Your exploration history will appear here
                </p>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>

      {/* Bottom decoration */}
      <div className="absolute inset-x-0 bottom-0 -z-10 transform-gpu overflow-hidden blur-3xl">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-blue-200 to-blue-400 opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
      </div>
    </div>
  );
}
