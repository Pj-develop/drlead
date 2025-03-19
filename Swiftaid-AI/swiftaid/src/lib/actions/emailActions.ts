"use server";

import { initializeEmailTracking, createTrackingPixel } from "../emailTracking";
import { sendEmail } from "@/lib/email";

export interface SendEmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
  campaign?: string;
  metadata?: Record<string, any>;
}

export async function sendTrackedEmail({
  to,
  subject,
  html,
  text,
  campaign,
  metadata = {},
}: SendEmailOptions) {
  try {
    // Generate a unique email ID
    const emailId = `${Date.now()}-${Math.random().toString(36).substring(2)}`;

    // Initialize tracking
    const token = await initializeEmailTracking(emailId, to, {
      subject,
      campaign,
      ...metadata,
    });

    // Get the base URL from environment variable or default to localhost
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";

    const trackingUrl = baseUrl + "/api/track/email?token=" + token;

    // Add tracking pixel to HTML content
    const trackingPixel = createTrackingPixel(token, trackingUrl);
    const htmlWithTracking = `${html}${trackingPixel}`;

    // TODO: Replace this with your email sending logic
    // For example, using nodemailer, SendGrid, etc.
    await sendEmail({
      to,
      subject,
      html: htmlWithTracking,
      text: text || "",
    });

    return {
      success: true,
      emailId,
      token,
    };
  } catch (error) {
    console.error("Error sending tracked email:", error);
    throw error;
  }
}
