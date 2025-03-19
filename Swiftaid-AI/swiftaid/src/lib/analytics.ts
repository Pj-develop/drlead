// Google Analytics 4 Event Tracking
export const GA_TRACKING_ID = "G-H6G87GL28Q";

// Page Views
export const pageview = (url: string) => {
  if (typeof window.gtag === "function") {
    window.gtag("config", GA_TRACKING_ID, {
      page_path: url,
    });
  }
};

// Events
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

// User Actions
export const trackUserAction = (action: string, details: any) => {
  event({
    action,
    category: "User Interaction",
    label: JSON.stringify(details),
  });
};

// City Exploration Events
export const trackCityExploration = (
  city: string,
  action: string,
  details: any
) => {
  event({
    action: `City_${action}`,
    category: "City Exploration",
    label: JSON.stringify({ city, ...details }),
  });
};

// Service Interactions
export const trackServiceInteraction = (
  service: string,
  action: string,
  details: any
) => {
  event({
    action: `Service_${action}`,
    category: "Service Interaction",
    label: JSON.stringify({ service, ...details }),
  });
};

// Lead Generation Events
export const trackLeadGeneration = (source: string, details: any) => {
  event({
    action: "Lead_Generated",
    category: "Lead Generation",
    label: JSON.stringify({ source, ...details }),
  });
};

// Type declaration for gtag
declare global {
  interface Window {
    gtag: (
      command: "config" | "event",
      targetId: string,
      config?: Record<string, any>
    ) => void;
  }
}

// Custom Analytics System

interface PageView {
  path: string;
  timestamp: number;
  duration?: number;
}

interface UserInteraction {
  type: string;
  details: any;
  timestamp: number;
}

class Analytics {
  private static instance: Analytics;
  private pageViews: PageView[] = [];
  private interactions: UserInteraction[] = [];
  private currentPageStartTime: number | null = null;

  private constructor() {}

  static getInstance(): Analytics {
    if (!Analytics.instance) {
      Analytics.instance = new Analytics();
    }
    return Analytics.instance;
  }

  // Page View Tracking
  trackPageView(path: string) {
    // End previous page view if exists
    if (this.currentPageStartTime) {
      const duration = Date.now() - this.currentPageStartTime;
      this.pageViews[this.pageViews.length - 1].duration = duration;
    }

    // Start new page view
    this.currentPageStartTime = Date.now();
    this.pageViews.push({
      path,
      timestamp: this.currentPageStartTime,
    });

    // Log to console in development
    if (process.env.NODE_ENV === "development") {
      console.log(`Page View: ${path}`);
    }
  }

  // User Interaction Tracking
  trackInteraction(type: string, details: any) {
    const interaction: UserInteraction = {
      type,
      details,
      timestamp: Date.now(),
    };
    this.interactions.push(interaction);

    // Log to console in development
    if (process.env.NODE_ENV === "development") {
      console.log(`Interaction: ${type}`, details);
    }
  }

  // Get Analytics Data
  getAnalyticsData() {
    return {
      pageViews: this.pageViews,
      interactions: this.interactions,
      totalPages: this.pageViews.length,
      totalInteractions: this.interactions.length,
      averagePageTime: this.calculateAveragePageTime(),
    };
  }

  // Calculate average page time
  private calculateAveragePageTime(): number {
    const validDurations = this.pageViews
      .map((view) => view.duration)
      .filter((duration): duration is number => duration !== undefined);

    if (validDurations.length === 0) return 0;

    return (
      validDurations.reduce((sum, duration) => sum + duration, 0) /
      validDurations.length
    );
  }

  // Clear Analytics Data
  clearData() {
    this.pageViews = [];
    this.interactions = [];
    this.currentPageStartTime = null;
  }
}

export const analytics = Analytics.getInstance();

// Export convenience functions
export const trackPageView = (path: string) => analytics.trackPageView(path);
export const trackInteraction = (type: string, details: any) =>
  analytics.trackInteraction(type, details);
export const getAnalyticsData = () => analytics.getAnalyticsData();
export const clearAnalyticsData = () => analytics.clearData();
