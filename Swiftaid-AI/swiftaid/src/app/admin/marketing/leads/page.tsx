"use client";

import { useState, useEffect } from "react";
import { format } from "date-fns";

interface Lead {
  rank: number;
  email: string;
  phone: string;
  score: number;
  current_city: string;
  target_city: string;
  budget: string;
  duration: string;
  status?: "new" | "contacted" | "qualified" | "converted" | "lost";
  notes?: string;
  last_contacted?: string;
  assigned_to?: string;
  foodPreferences?: string[];
  transportType?: string[];
  accommodationType?: string[];
  lifestyle?: string[];
  safety?: string;
}

export default function MarketingLeadsPage() {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);
  const [filter, setFilter] = useState<
    "all" | "new" | "contacted" | "qualified" | "converted" | "lost"
  >("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [isSendingEmail, setIsSendingEmail] = useState(false);
  const [emailStatus, setEmailStatus] = useState<{
    success?: boolean;
    message?: string;
    error?: string;
  } | null>(null);

  useEffect(() => {
    fetchLeads();
  }, []);

  const fetchLeads = async () => {
    try {
      const response = await fetch("/api/preferences/model");
      if (!response.ok) {
        throw new Error("Failed to fetch leads");
      }
      const data = await response.json();
      if (data.success && data.rankings) {
        const leadsWithStatus = data.rankings.map((lead: Lead) => ({
          ...lead,
          status: "new",
            notes: "",
          }))
          .filter((lead: Lead) => lead.score > 0);
        setLeads(leadsWithStatus);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const updateLeadStatus = async (lead: Lead, newStatus: Lead["status"]) => {
    try {
      setLeads(
        leads.map((l) =>
          l.email === lead.email
            ? {
                ...l,
                status: newStatus,
                last_contacted: new Date().toISOString(),
              }
            : l
        )
      );
    } catch (err: any) {
      setError(err.message);
    }
  };

  const updateLeadNotes = async (lead: Lead, notes: string) => {
    try {
      setLeads(
        leads.map((l) => (l.email === lead.email ? { ...l, notes } : l))
      );
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleSendEmail = async (lead: Lead) => {
    setIsSendingEmail(true);
    setEmailStatus(null);

    try {
      const response = await fetch("/api/marketing/send-email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: lead.email,
          currentCity: lead.current_city,
          targetCity: lead.target_city,
          budget: lead.budget,
          duration: lead.duration,
          preferences: {
            foodPreferences: lead.foodPreferences,
            transportType: lead.transportType,
            accommodationType: lead.accommodationType,
            lifestyle: lead.lifestyle,
            safety: lead.safety,
          },
        }),
      });

      const data = await response.json();

      if (data.success) {
        setEmailStatus({
          success: true,
          message: "Email sent successfully!",
        });
        updateLeadStatus(lead, "contacted");
      } else {
        throw new Error(data.error || "Failed to send email");
      }
    } catch (error: any) {
      setEmailStatus({
        success: false,
        error: error.message,
      });
    } finally {
      setIsSendingEmail(false);
    }
  };

  const filteredLeads = leads.filter((lead) => {
    const matchesFilter = filter === "all" || lead.status === filter;
    const matchesSearch =
      lead.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lead.phone.includes(searchTerm) ||
      lead.current_city.toLowerCase().includes(searchTerm.toLowerCase()) ||
      lead.target_city.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const getStatusColor = (status: Lead["status"]) => {
    switch (status) {
      case "new":
        return "bg-blue-100 text-blue-800";
      case "contacted":
        return "bg-yellow-100 text-yellow-800";
      case "qualified":
        return "bg-green-100 text-green-800";
      case "converted":
        return "bg-purple-100 text-purple-800";
      case "lost":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 text-black">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Marketing Leads</h1>
        <div className="flex gap-4">
          <select
            value={filter}
            onChange={(e) =>
              setFilter(e.target.value as Lead["status"] | "all")
            }
            className="rounded border-gray-300"
          >
            <option value="all">All Leads</option>
            <option value="new">New</option>
            <option value="contacted">Contacted</option>
            <option value="qualified">Qualified</option>
            <option value="converted">Converted</option>
            <option value="lost">Lost</option>
          </select>
          <input
            type="text"
            placeholder="Search leads..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="rounded border-gray-300"
          />
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Leads List */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Rank
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Contact
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Cities
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredLeads.map((lead) => (
                      <tr
                        key={lead.email}
                        onClick={() => setSelectedLead(lead)}
                        className="hover:bg-gray-50 cursor-pointer"
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          #{lead.rank}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {lead.score.toFixed(0)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <div>{lead.email}</div>
                          <div className="text-xs text-gray-400">
                            {lead.phone}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <div>
                            {lead.current_city} → {lead.target_city}
                          </div>
                          <div className="text-xs text-gray-400">
                            {lead.budget} • {lead.duration}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(
                              lead.status
                            )}`}
                          >
                            {lead.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Lead Details */}
          <div className="lg:col-span-1">
            {selectedLead ? (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4">Lead Details</h2>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium text-gray-700">
                      Contact Information
                    </h3>
                    <p className="text-sm text-gray-600">
                      {selectedLead.email}
                    </p>
                    <p className="text-sm text-gray-600">
                      {selectedLead.phone}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-700">Location</h3>
                    <p className="text-sm text-gray-600">
                      From: {selectedLead.current_city}
                    </p>
                    <p className="text-sm text-gray-600">
                      To: {selectedLead.target_city}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-700">Preferences</h3>
                    <p className="text-sm text-gray-600">
                      Budget: {selectedLead.budget}
                    </p>
                    <p className="text-sm text-gray-600">
                      Duration: {selectedLead.duration}
                    </p>
                    <p className="text-sm text-gray-600">
                      Score: {selectedLead.score.toFixed(0)}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-700">Status</h3>
                    <select
                      value={selectedLead.status}
                      onChange={(e) =>
                        updateLeadStatus(
                          selectedLead,
                          e.target.value as Lead["status"]
                        )
                      }
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    >
                      <option value="new">New</option>
                      <option value="contacted">Contacted</option>
                      <option value="qualified">Qualified</option>
                      <option value="converted">Converted</option>
                      <option value="lost">Lost</option>
                    </select>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-700">Notes</h3>
                    <textarea
                      value={selectedLead.notes}
                      onChange={(e) =>
                        updateLeadNotes(selectedLead, e.target.value)
                      }
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      rows={4}
                      placeholder="Add notes about this lead..."
                    />
                  </div>

                  <div className="flex gap-4">
                    <button
                      onClick={() => handleSendEmail(selectedLead)}
                      disabled={
                        isSendingEmail || selectedLead.status === "contacted"
                      }
                      className={`flex-1 px-4 py-2 rounded-md text-white ${
                        isSendingEmail || selectedLead.status === "contacted"
                          ? "bg-gray-400 cursor-not-allowed"
                          : "bg-blue-500 hover:bg-blue-600"
                      } flex items-center justify-center gap-2`}
                    >
                      {isSendingEmail ? (
                        <>
                          <svg
                            className="animate-spin h-5 w-5"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                              fill="none"
                            />
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            />
                          </svg>
                          Sending...
                        </>
                      ) : selectedLead.status === "contacted" ? (
                        "Already Contacted"
                      ) : (
                        "Send Marketing Email"
                      )}
                    </button>
                  </div>

                  {emailStatus && (
                    <div
                      className={`p-3 rounded-md ${
                        emailStatus.success
                          ? "bg-green-100 text-green-700"
                          : "bg-red-100 text-red-700"
                      }`}
                    >
                      {emailStatus.success
                        ? emailStatus.message
                        : emailStatus.error}
                    </div>
                  )}

                  {selectedLead.last_contacted && (
                    <div className="text-sm text-gray-500">
                      Last contacted:{" "}
                      {format(new Date(selectedLead.last_contacted), "PPp")}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md p-6 text-center text-gray-500">
                Select a lead to view details
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
