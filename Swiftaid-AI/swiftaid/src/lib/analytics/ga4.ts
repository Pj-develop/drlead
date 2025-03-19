// Google Analytics 4 Configuration
export const GA_TRACKING_ID = process.env.NEXT_PUBLIC_GA_ID || "";

// Pageview tracking
export const pageview = (url: string) => {
  if (typeof window.gtag === "function") {
    window.gtag("config", GA_TRACKING_ID, {
      page_path: url,
    });
  }
};

// Event tracking
export const event = ({
  action,
  category,
  label,
  value,
}: {
  action: string;
  category: string;
  label: string;
  value?: number;
}) => {
  if (typeof window.gtag === "function") {
    window.gtag("event", action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
};

// Lead tracking
export const trackLead = (data: {
  source: string;
  type: string;
  details: Record<string, any>;
}) => {
  if (typeof window.gtag === "function") {
    window.gtag("event", "lead_capture", {
      lead_source: data.source,
      lead_type: data.type,
      ...data.details,
    });
  }
};

// Service interest tracking
export const trackServiceInterest = (data: {
  service: string;
  interest_level: "high" | "medium" | "low";
  details: Record<string, any>;
}) => {
  if (typeof window.gtag === "function") {
    window.gtag("event", "service_interest", {
      service_name: data.service,
      interest_level: data.interest_level,
      ...data.details,
    });
  }
};
