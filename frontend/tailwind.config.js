/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#FAFAF7",
        "paper-muted": "#F4F4F0",
        "paper-card": "#FFFFFF",
        ink: "#111827",
        "ink-muted": "#6B7280",
        rule: "#E5E7EB",
        "rule-dark": "#D1D5DB",
        illicit: "#B42318",
        licit: "#1D4ED8",
        unknown: "#6B7280",
        accent: "#0F766E",
        "accent-hover": "#115E59",
        "accent-light": "#F0FDFA",
      },
      fontFamily: {
        serif: [
          '"Source Serif 4"',
          'Fraunces',
          'Georgia',
          'Cambria',
          '"Times New Roman"',
          'Times',
          'serif',
        ],
        sans: [
          'Inter',
          '"IBM Plex Sans"',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          '"IBM Plex Mono"',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Courier New"',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
};
