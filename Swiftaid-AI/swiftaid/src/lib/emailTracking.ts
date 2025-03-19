import { ObjectId, Document } from "mongodb";
import { connectToDatabase } from "./mongodb";
import crypto from "crypto";

export interface EmailTrackingData {
  _id?: ObjectId;
  emailId: string;
  recipientEmail: string;
  token: string;
  sentAt: Date;
  openedAt?: Date;
  openCount: number;
  metadata?: {
    subject?: string;
    campaign?: string;
    [key: string]: any;
  };
}

interface EmailTrackingStats {
  totalSent: number;
  totalOpened: number;
  openRate: number;
}

/**
 * Generates a unique tracking token for an email
 */
export function generateTrackingToken(
  emailId: string,
  recipientEmail: string
): string {
  const data = `${emailId}:${recipientEmail}:${Date.now()}`;
  return crypto.createHash("sha256").update(data).digest("hex");
}

/**
 * Creates a tracking URL for the email
 */
export function createTrackingUrl(token: string, baseUrl: string): string {
  const url = new URL("/api/track/email", baseUrl);
  url.searchParams.set("token", token);
  return url.toString();
}

/**
 * Creates a tracking pixel HTML string
 */
export function createTrackingPixel(token: string, baseUrl: string): string {
  const trackingUrl = createTrackingUrl(token, baseUrl);
  return `<img src="${trackingUrl}" width="1" height="1" alt="" style="display:none;"/>`;
}

/**
 * Initializes email tracking for a new email
 */
export async function initializeEmailTracking(
  emailId: string,
  recipientEmail: string,
  metadata?: EmailTrackingData["metadata"]
): Promise<string> {
  const { db } = await connectToDatabase();

  const token = generateTrackingToken(emailId, recipientEmail);

  const trackingData: EmailTrackingData = {
    emailId,
    recipientEmail,
    token,
    sentAt: new Date(),
    openCount: 0,
    metadata,
  };

  await db.collection("email_tracking").insertOne(trackingData);

  return token;
}

/**
 * Records an email open event
 */
export async function trackEmailOpen(token: string): Promise<void> {
  const { db } = await connectToDatabase();

  const now = new Date();

  await db.collection("email_tracking").updateOne(
    { token },
    {
      $set: {
        openedAt: now,
      },
      $inc: {
        openCount: 1,
      },
    }
  );
}

/**
 * Gets tracking data for an email
 */
export async function getEmailTrackingData(
  token: string
): Promise<EmailTrackingData | null> {
  const { db } = await connectToDatabase();

  const result = await db
    .collection<EmailTrackingData>("email_tracking")
    .findOne({ token });
  return result;
}

/**
 * Gets tracking statistics for multiple emails
 */
export async function getEmailTrackingStats(
  emailIds: string[]
): Promise<EmailTrackingStats> {
  const { db } = await connectToDatabase();

  const stats = await db
    .collection("email_tracking")
    .aggregate<EmailTrackingStats>([
      {
        $match: {
          emailId: { $in: emailIds },
        },
      },
      {
        $group: {
          _id: null,
          totalSent: { $sum: 1 },
          totalOpened: {
            $sum: {
              $cond: [{ $gt: ["$openCount", 0] }, 1, 0],
            },
          },
        },
      },
      {
        $project: {
          _id: 0,
          totalSent: 1,
          totalOpened: 1,
          openRate: {
            $cond: [
              { $eq: ["$totalSent", 0] },
              0,
              { $multiply: [{ $divide: ["$totalOpened", "$totalSent"] }, 100] },
            ],
          },
        },
      },
    ])
    .toArray();

  return stats[0] || { totalSent: 0, totalOpened: 0, openRate: 0 };
}

export async function getEmailsStatus() {
  const { db } = await connectToDatabase();
  return db.collection("email_tracking").find({}).toArray();
}
