import { resolve } from 'node:path';
import viteReact from '@vitejs/plugin-react-swc';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [viteReact()],
  server: {
    port: 5173,
    proxy: {
      '/token': 'http://localhost:8080',
      '/stream': 'http://localhost:8080',
      '/hls': {
        target: 'http://localhost:9000',
        // Mirrors the Caddyfile's /hls -> /livekit-hls rewrite so local `npm run dev`
        // hits the same bucket path as the in-cluster proxy.
        rewrite: (path) => path.replace(/^\/hls/, '/livekit-hls'),
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
});
