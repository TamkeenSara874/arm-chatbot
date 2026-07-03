import typography from '@tailwindcss/typography';
import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        aio: {
          50: '#FEF2EE',
          100: '#FDDED6',
          200: '#FABAAB',
          300: '#F68E7A',
          400: '#EE6A51',
          500: '#E85D3C',
          600: '#D44226',
          700: '#B0351C',
          800: '#912D1A',
          900: '#782819',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        'slide-in': {
          from: { transform: 'translateX(100%)' },
          to: { transform: 'translateX(0)' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'dot-bounce': {
          '0%, 80%, 100%': { transform: 'scale(0)' },
          '40%': { transform: 'scale(1)' },
        },
      },
      animation: {
        blink: 'blink 1s step-end infinite',
        'slide-in': 'slide-in 0.2s ease-out',
        'fade-in': 'fade-in 0.15s ease-out',
        'dot-1': 'dot-bounce 1.4s ease-in-out infinite',
        'dot-2': 'dot-bounce 1.4s ease-in-out 0.16s infinite',
        'dot-3': 'dot-bounce 1.4s ease-in-out 0.32s infinite',
      },
    },
  },
  plugins: [typography],
} satisfies Config;
