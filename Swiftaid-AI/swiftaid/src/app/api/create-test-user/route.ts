import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { connectToDatabase } from "@/lib/mongodb";

export async function POST() {
  try {
    const { db } = await connectToDatabase();
    const hashedPassword = await bcrypt.hash("test123", 10);

    const result = await db.collection("users").insertOne({
      email: "test@example.com",
      password: hashedPassword,
      name: "Test User",
    });

    return NextResponse.json({
      success: true,
      message: "Test user created successfully",
      result,
    });
  } catch (error) {
    console.error("Error creating test user:", error);
    return NextResponse.json(
      { success: false, message: "Error creating test user" },
      { status: 500 }
    );
  }
}
