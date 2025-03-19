import { GoogleGenerativeAI } from "@google/generative-ai";
import { LeadEmailData } from "@/types/email";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

export async function generatePersonalizedEmail(leadData: LeadEmailData) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const prompt = `
    Generate a personalized marketing email for a potential customer with the following details in html format, no extra text inside \`\`\`html\`\`\` tags:
    
    Current City: ${leadData.currentCity}
    Target City: ${leadData.targetCity}
    Budget: ${leadData.budget}
    Duration: ${leadData.duration}
    Customer Name: Customer
    Preferences:
    - Food: ${
      leadData.preferences.foodPreferences?.join(", ") || "Not specified"
    }
    - Transport: ${
      leadData.preferences.transportType?.join(", ") || "Not specified"
    }
    - Accommodation: ${
      leadData.preferences.accommodationType?.join(", ") || "Not specified"
    }
    - Lifestyle: ${
      leadData.preferences.lifestyle?.join(", ") || "Not specified"
    }
    - Safety: ${leadData.preferences.safety || "Not specified"}

    My company is SwiftAid, a travel and relocation company.
    My name is Mokshit Jain.
    Contact me at mokshitjain18@gmail.com, +91 9876543210
    Please create:
    1. An engaging subject line
    2. A personalized email body that:
       - Acknowledges their specific preferences
       - Highlights relevant services for their trip/relocation (see duration and budget)
       - Includes specific recommendations based on their preferences
       - Maintains a professional yet friendly tone
       - Includes a clear call to action
  `;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    // console.log(text);
    // Extract HTML content between ```html and ``` markers
    const htmlMatch = text.match(/```html\n([\s\S]*?)\n```/);
    const emailContent = htmlMatch ? htmlMatch[1] : text;

    // Extract subject line
    const subjectMatch = text.match(/Subject: (.*?)(?:\n|$)/);
    const subject = subjectMatch
      ? subjectMatch[1]
      : `Offers for your upcoming move to ${leadData.targetCity}`;

    return {
      subject,
      body: emailContent,
    };
  } catch (error) {
    console.error("Error generating email content:", error);
    throw error;
  }
}
