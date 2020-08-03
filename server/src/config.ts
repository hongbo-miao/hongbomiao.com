const { nodeEnv, port } = process.env;

const Config = {
  nodeEnv,
  port: port || 3001,

  devAllowList: ['http://localhost:3000', 'http://localhost:3001'],
  prodAllowList: ['https://hongbomiao.com', 'https://www.hongbomiao.com'],

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: nodeEnv,
  },
};

export default Config;
