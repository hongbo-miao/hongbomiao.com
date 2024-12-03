import path from 'path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [
          '@babel/plugin-transform-react-display-name',
          '@babel/plugin-transform-private-property-in-object',
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      'src': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000,
    host: true, // needed for docker
  },
  build: {
    outDir: 'build',
    sourcemap: true,
  },
  publicDir: 'public',
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: [
        'node_modules/',
        'src/setupTests.ts',
        '**/*.query.ts',
        '**/*.story.tsx',
        '**/*.type.ts',
        'src/shared/libs/*',
        'src/shared/utils/initSentry.ts',
      ],
    },
  },
});
