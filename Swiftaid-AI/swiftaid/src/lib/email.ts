import nodemailer from "nodemailer";
import { EmailData } from "@/types/email";

// Create a transporter using SMTP
const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: Number(process.env.SMTP_PORT),
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

export async function sendEmail(emailData: EmailData) {
  try {
    const info = await transporter.sendMail({
      from: `"SwiftAid" <${process.env.SMTP_USER}>`,
      to: emailData.to,
      subject: emailData.subject,
      text: emailData.text || "",
      html: emailData.html || "",
    });

    console.log("Email sent:", info.messageId);
    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error("Error sending email:", error);
    throw error;
  }
}
