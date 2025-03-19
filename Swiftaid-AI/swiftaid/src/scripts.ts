import bcrypt from "bcryptjs";
import { connectToDatabase } from "./lib/mongodb";

async function createTestUser() {
  try {
    const { db } = await connectToDatabase();
    const hashedPassword = await bcrypt.hash("test123", 10);

    const result = await db.collection("users").insertOne({
      email: "test@example.com",
      password: hashedPassword,
      name: "Test User",
    });

    console.log("Test user created successfully:", result);
  } catch (error) {
    console.error("Error creating test user:", error);
  } finally {
    process.exit(0);
  }
}

createTestUser();
