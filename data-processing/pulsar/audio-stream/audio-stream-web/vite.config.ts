import { resolve } from 'node:path';
import viteReact from '@vitejs/plugin-react-swc';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [viteReact()],
  optimizeDeps: {
    exclude: ['@shiguredo/rnnoise-wasm'],
  },
  server: {
    port: 5173,
    proxy: {
      '/token': 'http://localhost:8080',
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
