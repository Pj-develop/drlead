import { PreferencesFormData } from "@/types/preferences";

export function convertToCSV(data: PreferencesFormData[]) {
  // Define headers based on the PreferencesFormData structure
  const headers = [
    "Email",
    "Phone",
    "Income",
    "Current City",
    "Budget",
    "Food Preferences",
    "Target City",
    "Distance",
    "Start Date",
    "Duration",
    "Transport Types",
    "Accommodation Types",
    "Lifestyle",
    "Safety Priority",
  ];

  // Create CSV header row
  let csv = headers.join(",") + "\n";

  // Add data rows
  data.forEach((item) => {
    const row = [
      `"${item.email}"`,
      `"${item.phone}"`,
      `"${item.income}"`,
      `"${item.currentCity}"`,
      `"${item.budget}"`,
      `"${item.foodPreferences.join("; ")}"`,
      `"${item.targetCity}"`,
      `"${item.distance}"`,
      `"${item.startDate}"`,
      `"${item.duration}"`,
      `"${item.transportType.join("; ")}"`,
      `"${item.accommodationType.join("; ")}"`,
      `"${item.lifestyle}"`,
      `"${item.safety}"`,
    ];
    csv += row.join(",") + "\n";
  });

  return csv;
}
