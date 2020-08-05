const { NODE_ENV, PORT } = process.env;

const Config = {
  nodeEnv: NODE_ENV,
  port: PORT || 3001,

  devCORSAllowList: [
    'electron://altair', // Altair GraphQL Client
    'https://localhost:3000',
    'https://localhost:3001',
    'null', // Safari reports CSP violation on localhost
  ],
  prodCORSAllowList: ['https://hongbomiao.com', 'https://www.hongbomiao.com'],

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default Config;
