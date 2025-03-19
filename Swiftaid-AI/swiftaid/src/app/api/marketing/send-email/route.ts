import { NextResponse } from "next/server";
import { generatePersonalizedEmail } from "@/lib/gemini";
import { LeadEmailData } from "@/types/email";
import { sendTrackedEmail } from "@/lib/actions/emailActions";

export async function POST(request: Request) {
  try {
    const leadData: LeadEmailData = await request.json();

    // Generate personalized email content
    const emailContent = await generatePersonalizedEmail(leadData);
    // console.log(emailContent);
    // Send the email
    const result = await sendTrackedEmail({
      to: leadData.email,
      subject: emailContent.subject,
      html: emailContent.body,
      text: emailContent.body,
    });

    return NextResponse.json({
      success: true,
      message: "Email sent successfully",
      messageId: result.emailId,
    });
  } catch (error: any) {
    console.error("Error sending marketing email:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || "Failed to send email",
      },
      { status: 500 }
    );
  }
}
