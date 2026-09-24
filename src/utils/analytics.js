/**
 * Analytics Utility for Google Analytics 4 (GA4)
 * Only sends data in production; logs to console during development.
 */

const GA_ID = import.meta.env.VITE_GA_MEASUREMENT_ID;

const isLocalhost = () => {
  if (typeof window === 'undefined') return true;
  return (
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === '' ||
    import.meta.env.DEV
  );
};

/**
 * Track a pageview within the SPA
 * @param {string} pageTitle - Name of the view/page
 * @param {string} [path] - Virtual path for GA reports
 */
export const trackPageView = (pageTitle, path) => {
  if (typeof window === 'undefined') return;

  const pagePath =
    path ||
    (pageTitle
      ? `/${pageTitle.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
      : window.location.pathname);

  if (isLocalhost()) {
    console.log('[GA4 Dev PageView]:', pageTitle, pagePath);
    return;
  }

  if (typeof window.gtag === 'function' && GA_ID) {
    window.gtag('event', 'page_view', {
      page_title: pageTitle,
      page_location: window.location.href,
      page_path: pagePath,
      send_to: GA_ID
    });
  }
};

/**
 * Track a custom user interaction event in GA4
 * @param {string} eventName - Name of the event (e.g. 'guest_explore', 'ai_plan_generated')
 * @param {Object} [params] - Key-value metadata for the event
 */
export const trackEvent = (eventName, params = {}) => {
  if (typeof window === 'undefined') return;

  if (isLocalhost()) {
    console.log('[GA4 Dev Event]:', eventName, params);
    return;
  }

  if (typeof window.gtag === 'function' && GA_ID) {
    window.gtag('event', eventName, {
      ...params,
      send_to: GA_ID
    });
  }
};

