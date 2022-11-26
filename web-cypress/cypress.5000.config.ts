import { defineConfig } from 'cypress';

export default defineConfig({
  env: {
    domain: 'http://localhost',
    webPort: '5000',
    serverPort: '5000',
  },
  video: false,
  e2e: {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
  },
});
