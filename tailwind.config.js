/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        finance: {
          bg: '#f8fafc',
          surface: '#ffffff',
          'text-primary': '#0f172a',
          'text-secondary': '#475569',
          border: '#e2e8f0',
          'table-header-bg': '#f1f5f9',
          'table-header-text': '#475569',
          primary: '#059669',
          'primary-soft': 'rgba(16, 185, 129, 0.1)',
          teal: '#0d9488',
          positive: '#059669',
          negative: '#dc2626',
          warning: '#d97706',
          'score-bg': '#f0fdf4',
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
