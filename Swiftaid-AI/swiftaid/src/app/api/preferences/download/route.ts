import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { connectToDatabase } from "@/lib/mongodb";
import { convertToCSV } from "@/utils/csvGenerator";

export async function GET() {
  try {
    // Check authentication
    const session = await getServerSession();
    if (!session) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // Connect to MongoDB
    const { db } = await connectToDatabase();

    // Fetch all preferences data
    const preferences = await db.collection("preferences").find({}).toArray();

    // Convert data to CSV
    const csv = convertToCSV(preferences as any);

    // Set headers for file download
    const headers = {
      "Content-Type": "text/csv",
      "Content-Disposition": "attachment; filename=preferences.csv",
    };

    // Return CSV file
    return new NextResponse(csv, { headers });
  } catch (error) {
    console.error("Error downloading preferences:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
