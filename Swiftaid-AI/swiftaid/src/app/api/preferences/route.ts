import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import clientPromise from "@/lib/mongodb";
import { ObjectId } from "mongodb";

export async function POST(req: Request) {
  try {
    const session = await getServerSession();
    if (!session?.user?.email) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const data = await req.json();
    const client = await clientPromise;
    const db = client.db("swiftaid");
    const collection = db.collection("preferences");

    // Update or insert preferences
    const result = await collection.insertOne(
      {
        ...data,
        userEmail: session.user.email,
        updatedAt: new Date(),
      },
    );

    return NextResponse.json({ success: true, result });
  } catch (error) {
    console.error("Error saving preferences:", error);
    return NextResponse.json(
      { error: "Failed to save preferences" },
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
    const collection = db.collection("preferences");

    const preferences = await collection.findOne({
      userEmail: session.user.email,
    });

    return NextResponse.json(preferences || {});
  } catch (error) {
    console.error("Error fetching preferences:", error);
    return NextResponse.json(
      { error: "Failed to fetch preferences" },
      { status: 500 }
    );
  }
}
