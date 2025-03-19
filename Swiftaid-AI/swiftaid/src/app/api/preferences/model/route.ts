import { NextResponse } from "next/server";
import { connectToDatabase } from "@/lib/mongodb";

export async function GET() {
  try {
    // Connect to MongoDB
    const { db } = await connectToDatabase();

    // Fetch all preferences
    const preferences = await db.collection("preferences").find({}).toArray();

    // Call the Python model API
    const modelResponse = await fetch("http://127.0.0.1:5000/rank_customers", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(preferences),
    });

    if (!modelResponse.ok) {
      throw new Error("Failed to get model analysis");
    }
    // console.log(modelResponse);
    const modelData = await modelResponse.json();

    return NextResponse.json(modelData);
  } catch (error: any) {
    console.error("Error in model analysis:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || "Failed to analyze preferences",
      },
      { status: 500 }
    );
  }
}
