import { defineConfig } from 'cypress';

export default defineConfig({
  env: {
    domain: 'http://localhost',
    webPort: '5000',
    serverPort: '5000',
  },
  video: false,
});
