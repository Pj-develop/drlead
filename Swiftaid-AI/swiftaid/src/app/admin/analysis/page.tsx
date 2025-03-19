"use client";

import { useState } from "react";

const ScoreBreakdown = () => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-xl font-semibold mb-4">
        Score Calculation Breakdown
      </h3>
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium mb-2">Total Score: 100 points</h4>
            <ul className="space-y-2">
              <li className="flex justify-between">
                <span>Income</span>
                <span className="font-medium">20 points</span>
              </li>
              <li className="flex justify-between">
                <span>Budget</span>
                <span className="font-medium">15 points</span>
              </li>
              <li className="flex justify-between">
                <span>Duration</span>
                <span className="font-medium">20 points</span>
              </li>
              <li className="flex justify-between">
                <span>Safety</span>
                <span className="font-medium">10 points</span>
              </li>
              <li className="flex justify-between">
                <span>Distance</span>
                <span className="font-medium">10 points</span>
              </li>
              <li className="flex justify-between">
                <span>Start Date</span>
                <span className="font-medium">10 points</span>
              </li>
              <li className="flex justify-between">
                <span>Preferences</span>
                <span className="font-medium">15 points</span>
              </li>
            </ul>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium mb-2">Value Mappings</h4>
            <div className="space-y-3">
              <div>
                <p className="font-medium">Income Ranges:</p>
                <ul className="text-sm">
                  <li>0-30000 → 15000</li>
                  <li>30000-50000 → 40000</li>
                  <li>50000-100000 → 75000</li>
                  <li>100000+ → 150000</li>
                </ul>
              </div>
              <div>
                <p className="font-medium">Budget Levels:</p>
                <ul className="text-sm">
                  <li>low → 1</li>
                  <li>medium → 2</li>
                  <li>high → 3</li>
                </ul>
              </div>
              <div>
                <p className="font-medium">Duration:</p>
                <ul className="text-sm">
                  <li>0-3 months → 1.5</li>
                  <li>3-6 months → 4.5</li>
                  <li>6-12 months → 9</li>
                  <li>12+ months → 15</li>
                  <li>permanent → 24</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-medium mb-2">Scoring Rules</h4>
          <ul className="space-y-2 text-sm">
            <li>• Higher income = higher score (0-20 points)</li>
            <li>• Higher budget = higher score (0-15 points)</li>
            <li>• Lower distance = higher score (0-10 points)</li>
            <li>• Longer duration = higher score (0-20 points)</li>
            <li>• Higher safety = higher score (0-10 points)</li>
            <li>• Sooner start = higher score (0-10 points)</li>
            <li>• More preferences = higher score (0-15 points)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default function AnalysisPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [modelAnalysis, setModelAnalysis] = useState<any>(null);

  const handleGenerateAnalysis = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch("/api/preferences/analyze");

      if (!response.ok) {
        throw new Error("Failed to generate analysis");
      }

      const data = await response.json();
      setAnalysis(data.analysis);
    } catch (error) {
      setError("Failed to generate analysis. Please try again.");
      console.error("Error generating analysis:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelAnalysis = async () => {
    try {
      setIsModelLoading(true);
      setError(null);

      const response = await fetch("/api/preferences/model");

      if (!response.ok) {
        throw new Error("Failed to generate model analysis");
      }

      const data = await response.json();
      setModelAnalysis(data);
    } catch (error) {
      setError(
        "Failed to generate model analysis. Please ensure the model server is running."
      );
      console.error("Error generating model analysis:", error);
    } finally {
      setIsModelLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 text-black">
      <h1 className="text-3xl font-bold mb-8">Preferences Analysis</h1>

      <ScoreBreakdown />

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Lead Generation Analysis
        </h1>
        <p className="text-gray-600">
          Generate AI-powered insights from user preferences data.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      <div className="mb-8 flex gap-4">
        <button
          onClick={handleGenerateAnalysis}
          disabled={isLoading}
          className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Generating Analysis...
            </>
          ) : (
            "Generate Gemini Analysis"
          )}
        </button>

        <button
          onClick={handleModelAnalysis}
          disabled={isModelLoading}
          className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isModelLoading ? (
            <>
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Running Model Analysis...
            </>
          ) : (
            "Run Model Analysis"
          )}
        </button>
      </div>

      {analysis && (
        <div className="prose prose-blue max-w-none bg-white rounded-lg shadow-sm p-6 border border-gray-200 mb-8">
          <h2 className="text-xl font-semibold mb-4">Gemini Analysis</h2>
          <div
            dangerouslySetInnerHTML={{
              __html: analysis.replace(/\n/g, "<br/>"),
            }}
          />
        </div>
      )}

      {modelAnalysis && modelAnalysis.success && (
        <div className="prose prose-blue max-w-none bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Model Analysis</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Email
                  </th>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Phone
                  </th>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score (100)
                  </th>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current City
                  </th>
                  <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Target City
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {modelAnalysis.rankings.map((customer: any) => (
                  <tr key={customer.rank}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.rank}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.email}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.phone}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.score.toFixed(0)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.current_city}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {customer.target_city}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-sm text-gray-500">
            <p>
              Total Customers: {modelAnalysis.model_metrics.total_customers}
            </p>
            <p>
              Analysis Time:{" "}
              {new Date(modelAnalysis.model_metrics.timestamp).toLocaleString()}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
