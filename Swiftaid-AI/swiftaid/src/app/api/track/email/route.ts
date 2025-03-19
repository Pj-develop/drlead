import { NextResponse } from "next/server";
import {
  getEmailsStatus,
  getEmailTrackingData,
  trackEmailOpen,
} from "@/lib/emailTracking";

// Create a 1x1 transparent GIF
const TRACKING_PIXEL = Buffer.from(
  "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
  "base64"
);

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const token = searchParams.get("token");

    if (!token) {
      return new NextResponse("Invalid token", { status: 400 });
    }

    // Track the email open
    await trackEmailOpen(token);

    // Return the tracking pixel
    return new NextResponse(TRACKING_PIXEL, {
      headers: {
        "Content-Type": "image/gif",
        "Cache-Control":
          "no-store, no-cache, must-revalidate, proxy-revalidate",
        Pragma: "no-cache",
        Expires: "0",
      },
    });
  } catch (error) {
    console.error("Error tracking email open:", error);
    return new NextResponse("Error tracking email", { status: 500 });
  }
}

// Get the emails status
export async function POST() {
  try {
    // Track the email open
    const emailStatus = await getEmailsStatus();

    return new NextResponse(JSON.stringify(emailStatus));
  } catch (error) {
    console.error("Error tracking email open:", error);
    return new NextResponse("Error tracking email", { status: 500 });
  }
}
