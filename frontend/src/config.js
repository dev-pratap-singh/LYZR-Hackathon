/**
 * Application Configuration
 * Manages environment-specific settings for API URLs
 */

// API Base URL - uses environment variable in production, localhost in development
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Export for convenience
export default {
  API_URL,
};
