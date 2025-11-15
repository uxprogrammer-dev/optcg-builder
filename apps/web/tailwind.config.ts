import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{js,ts,jsx,tsx}', './components/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f5f3ff',
          100: '#edeafe',
          200: '#d9d4fd',
          300: '#bcb3f9',
          400: '#9a8af3',
          500: '#7d61ec',
          600: '#5f39dd',
          700: '#4a2ab1',
          800: '#3d248d',
          900: '#34226f',
        },
      },
    },
  },
  plugins: [],
};

export default config;
