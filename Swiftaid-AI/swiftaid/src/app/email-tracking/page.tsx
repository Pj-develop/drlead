"use server";

import { getEmailsStatus } from "@/lib/emailTracking";
import { format } from "date-fns";

export default async function EmailTrackingPage() {
  const trackingData = await getEmailsStatus();

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Email Tracking Analytics
            </h1>
            <p className="mt-2 text-sm text-gray-500">
              Track and analyze email engagement metrics
            </p>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Total Emails
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-gray-900">
                  {trackingData?.length || 0}
                </dd>
              </div>
            </div>
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Total Opens
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-gray-900">
                  {trackingData?.reduce(
                    (sum, email) => sum + email.openCount,
                    0
                  ) || 0}
                </dd>
              </div>
            </div>
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Opened Emails
                </dt>
                <dd className="mt-1 text-3xl font-semibold text-gray-900">
                  {trackingData?.filter((email) => email.openCount > 0)
                    .length || 0}
                </dd>
              </div>
            </div>
          </div>

          {/* Email Table */}
          {trackingData && trackingData.length > 0 ? (
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Recipient
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Subject
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Sent At
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Opens
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Last Opened
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trackingData.map((email) => (
                      <tr key={email.emailId}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {email.recipientEmail}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          {email.metadata?.subject?.replace("</p>", "")}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {format(new Date(email.sentAt), "MMM d, yyyy HH:mm")}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {email.openCount}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {email.openedAt
                            ? format(
                                new Date(email.openedAt),
                                "MMM d, yyyy HH:mm"
                              )
                            : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No tracking data available
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Start sending emails to see tracking information
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
