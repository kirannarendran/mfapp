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
          bg: '#0f172a',
          surface: '#1e293b',
          'text-primary': '#f8fafc',
          'text-secondary': '#94a3b8',
          border: '#334155',
          'table-header-bg': '#0f172a',
          'table-header-text': '#94a3b8',
          primary: '#0ea5e9',
          'primary-soft': 'rgba(14, 165, 233, 0.1)',
          teal: '#10b981',
          positive: '#10b981',
          negative: '#ef4444',
          warning: '#f59e0b',
          'score-bg': 'rgba(30, 41, 59, 0.5)',
        }
      }
    },
  },
  plugins: [],
}
