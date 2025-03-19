export interface EmailTemplate {
  subject: string;
  body: string;
  variables: string[];
}

export interface EmailData {
  to: string;
  subject: string;
  html: string;
  text: string;
}

export interface LeadEmailData {
  email: string;
  name?: string;
  currentCity: string;
  targetCity: string;
  budget: string;
  duration: string;
  preferences: {
    foodPreferences?: string[];
    transportType?: string[];
    accommodationType?: string[];
    lifestyle?: string[];
    safety?: string;
  };
}
