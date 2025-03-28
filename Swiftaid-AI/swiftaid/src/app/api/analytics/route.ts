import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import clientPromise from "@/lib/mongodb";

export async function POST(req: Request) {
  try {
    const session = await getServerSession();
    if (!session?.user?.email) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const data = await req.json();
    const client = await clientPromise;
    const db = client.db("swiftaid");
    const collection = db.collection("analytics");

    // Insert analytics event
    const result = await collection.insertOne({
      ...data,
      userEmail: session.user.email,
      timestamp: new Date(),
    });

    return NextResponse.json({ success: true, result });
  } catch (error) {
    console.error("Error saving analytics:", error);
    return NextResponse.json(
      { error: "Failed to save analytics" },
      { status: 500 }
    );
  }
}

export async function GET(req: Request) {
  try {
    const session = await getServerSession();
    if (!session?.user?.email) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const client = await clientPromise;
    const db = client.db("swiftaid");
    const collection = db.collection("analytics");

    // Get analytics data for the user
    const analytics = await collection
      .find({ userEmail: session.user.email })
      .sort({ timestamp: -1 })
      .limit(100)
      .toArray();

    return NextResponse.json(analytics);
  } catch (error) {
    console.error("Error fetching analytics:", error);
    return NextResponse.json(
      { error: "Failed to fetch analytics" },
      { status: 500 }
    );
  }
}
