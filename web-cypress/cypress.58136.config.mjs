import { defineConfig } from 'cypress';

export default defineConfig({
  env: {
    domain: 'http://localhost',
    webPort: '58136',
    serverPort: '58136',
  },
  video: false,
  e2e: {
    // eslint-disable-next-line no-unused-vars
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
  },
});
