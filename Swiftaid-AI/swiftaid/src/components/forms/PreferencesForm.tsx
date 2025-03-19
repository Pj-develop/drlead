"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface PreferencesFormData {
  email: string;
  phone: string;
  pagesVisited: string[];
  income: string;
  currentCity: string;
  budget: string;
  foodPreferences: string[];
  targetCity: string;
  distance: string;
  startDate: string;
  duration: string;
  transportType: string[];
  accommodationType: string[];
  lifestyle: string;
  safety: string;
}

export default function PreferencesForm() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<PreferencesFormData>({
    email: "",
    phone: "",
    pagesVisited: [],
    income: "",
    currentCity: "",
    budget: "",
    foodPreferences: [],
    targetCity: "",
    distance: "",
    startDate: "",
    duration: "",
    transportType: [],
    accommodationType: [],
    lifestyle: "",
    safety: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      setError(null);

      // console.log(formData);

      const response = await fetch("/api/preferences", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to save preferences");
      }

      router.push("/explore");
    } catch (error) {
      setError("Failed to save preferences. Please try again.");
      console.error("Error saving preferences:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    if (type === "checkbox") {
      const checkbox = e.target as HTMLInputElement;
      const currentValues = [
        ...(formData[name as keyof PreferencesFormData] as string[]),
      ];
      if (checkbox.checked) {
        currentValues.push(value);
      } else {
        const index = currentValues.indexOf(value);
        if (index > -1) {
          currentValues.splice(index, 1);
        }
      }
      setFormData({ ...formData, [name]: currentValues });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">

      <form onSubmit={handleSubmit} className="space-y-12">
        {error && <div className="text-sm text-red-600 mb-4">{error}</div>}

        <div className="space-y-8">
          {/* Contact Information */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Contact Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Email Address
                </label>
                <input
                  type="email"
                  name="email"
                  id="email"
                  required
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label
                  htmlFor="phone"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Phone Number
                </label>
                <input
                  type="tel"
                  name="phone"
                  id="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </section>

          {/* Financial Information */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Financial Information
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label
                  htmlFor="income"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Monthly Income Range
                </label>
                <select
                  id="income"
                  name="income"
                  value={formData.income}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select income range</option>
                  <option value="0-30000">₹0 - ₹30,000</option>
                  <option value="30000-50000">₹30,000 - ₹50,000</option>
                  <option value="50000-100000">₹50,000 - ₹1,00,000</option>
                  <option value="100000+">₹1,00,000+</option>
                </select>
              </div>
              <div>
                <label
                  htmlFor="budget"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Monthly Budget for City
                </label>
                <select
                  id="budget"
                  name="budget"
                  value={formData.budget}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select budget range</option>
                  <option value="low">Low (Under ₹15,000)</option>
                  <option value="medium">Medium (₹15,000 - ₹30,000)</option>
                  <option value="high">High (₹30,000+)</option>
                </select>
              </div>
            </div>
          </section>

          {/* Location Details */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Location Details
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label
                  htmlFor="currentCity"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Current City
                </label>
                <input
                  type="text"
                  name="currentCity"
                  id="currentCity"
                  value={formData.currentCity}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label
                  htmlFor="targetCity"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Target City
                </label>
                <input
                  type="text"
                  name="targetCity"
                  id="targetCity"
                  value={formData.targetCity}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label
                  htmlFor="distance"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Distance between Cities
                </label>
                <select
                  id="distance"
                  name="distance"
                  value={formData.distance}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select distance range</option>
                  <option value="0-100">0-100 km</option>
                  <option value="100-300">100-300 km</option>
                  <option value="300-500">300-500 km</option>
                  <option value="500-1000">500-1000 km</option>
                  <option value="1000+">1000+ km</option>
                </select>
              </div>
            </div>
          </section>

          {/* Stay Details */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Stay Details
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label
                  htmlFor="startDate"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Start Date
                </label>
                <input
                  type="date"
                  name="startDate"
                  id="startDate"
                  value={formData.startDate}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label
                  htmlFor="duration"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Duration
                </label>
                <select
                  id="duration"
                  name="duration"
                  value={formData.duration}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select duration</option>
                  <option value="1-3 months">1-3 months</option>
                  <option value="3-6 months">3-6 months</option>
                  <option value="6-12 months">6-12 months</option>
                  <option value="1+ year">1+ year</option>
                  <option value="permanent">Permanent Stay</option>
                </select>
              </div>
            </div>
          </section>

          {/* Transport & Accommodation */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Transport & Accommodation
            </h2>
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Preferred Transport Types
                </label>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { value: "public", label: "Public Transport" },
                    { value: "private", label: "Private Vehicle" },
                  ].map((option) => (
                    <label
                      key={option.value}
                      className="flex items-center space-x-3"
                    >
                      <input
                        type="checkbox"
                        name="transportType"
                        value={option.value}
                        checked={formData.transportType.includes(option.value)}
                        onChange={handleChange}
                        className="h-4 w-4 border-gray-300 rounded text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">
                        {option.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Accommodation Type
                </label>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { value: "apartment", label: "Apartment" },
                    { value: "house", label: "House" },
                    { value: "pg", label: "PG/Hostel" },
                    { value: "shared", label: "Shared Accommodation" },
                  ].map((option) => (
                    <label
                      key={option.value}
                      className="flex items-center space-x-3"
                    >
                      <input
                        type="checkbox"
                        name="accommodationType"
                        value={option.value}
                        checked={formData.accommodationType.includes(
                          option.value
                        )}
                        onChange={handleChange}
                        className="h-4 w-4 border-gray-300 rounded text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">
                        {option.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Food Preferences */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Food Preferences
            </h2>
            <div className="grid grid-cols-2 gap-4">
              {[
                { value: "vegetarian", label: "Vegetarian" },
                { value: "non-vegetarian", label: "Non-Vegetarian" },
                { value: "vegan", label: "Vegan" },
                { value: "jain", label: "Jain Food" },
              ].map((option) => (
                <label
                  key={option.value}
                  className="flex items-center space-x-3"
                >
                  <input
                    type="checkbox"
                    name="foodPreferences"
                    value={option.value}
                    checked={formData.foodPreferences.includes(option.value)}
                    onChange={handleChange}
                    className="h-4 w-4 border-gray-300 rounded text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">{option.label}</span>
                </label>
              ))}
            </div>
          </section>

          {/* Additional Preferences */}
          <section>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Additional Preferences
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label
                  htmlFor="lifestyle"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Lifestyle Preference
                </label>
                <select
                  id="lifestyle"
                  name="lifestyle"
                  value={formData.lifestyle}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select lifestyle</option>
                  <option value="quiet">Quiet and Peaceful</option>
                  <option value="active">Active and Vibrant</option>
                  <option value="balanced">Balanced</option>
                </select>
              </div>
              <div>
                <label
                  htmlFor="safety"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Safety Priority
                </label>
                <select
                  id="safety"
                  name="safety"
                  value={formData.safety}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select safety priority</option>
                  <option value="high">High Priority</option>
                  <option value="medium">Medium Priority</option>
                  <option value="low">Low Priority</option>
                </select>
              </div>
            </div>
          </section>
        </div>

        <div className="mt-8">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full md:w-auto px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "Saving..." : "Save Preferences"}
          </button>
        </div>
      </form>
    </div>
  );
}
