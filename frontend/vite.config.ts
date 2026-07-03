import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// loadEnv (not process.env) avoids needing @types/node just to read one
// build-time variable in this config file.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  // Overridable so the dev proxy can reach the backend by its Docker Compose
  // service name ("http://backend:8000") when this runs inside the frontend
  // container, while still defaulting to localhost for the existing
  // run-it-yourself-with-npm-run-dev workflow.
  const apiProxyTarget = env.VITE_API_PROXY_TARGET || 'http://localhost:8000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  };
});
