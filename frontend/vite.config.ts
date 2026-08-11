import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The dev server proxies to the API so the client runs same-origin in development,
// matching how it is deployed behind one host in production (see infrastructure/).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});
