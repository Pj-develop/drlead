import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { connectToDatabase } from "@/lib/mongodb";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

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

    // Create prompt for Gemini
    const prompt = `As a lead generation expert, analyze this user preference data. Please do not miss any data:
    ${JSON.stringify(preferences, null)}
    Score Calculation Breakdown
    Total Score: 100 points
    Income
    20 points
    Budget
    15 points
    Duration
    20 points
    Safety
    10 points
    Distance
    10 points
    Start Date
    10 points
    Preferences
    15 points
    Value Mappings
    Income Ranges:

    0-30000 → 15000
    30000-50000 → 40000
    50000-100000 → 75000
    100000+ → 150000
    Budget Levels:

    low → 1
    medium → 2
    high → 3
    Duration:

    0-3 months → 1.5
    3-6 months → 4.5
    6-12 months → 9
    12+ months → 15
    permanent → 24
    Scoring Rules
    • Higher income = higher score (0-20 points)
    • Higher budget = higher score (0-15 points)
    • Lower distance = higher score (0-10 points)
    • Longer duration = higher score (0-20 points)
    • Higher safety = higher score (0-10 points)
    • Sooner start = higher score (0-10 points)
    • More preferences = higher score (0-15 points)

    Please provide data in tabular format:
    1. Most likely to become customers within a month (user email and score) for all users
    2. Most likely to become customers after 3 months
    3. Most Liked cities

    Format the response in tabular format. No extra text or comments.`;

    // Get Gemini model
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    // Generate analysis
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const analysis = response.text();

    return NextResponse.json({ analysis });
  } catch (error) {
    console.error("Error analyzing preferences:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
