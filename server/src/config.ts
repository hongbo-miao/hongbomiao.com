const Config = {
  nodeEnv: process.env.NODE_ENV,
  port: process.env.PORT,

  sentryDSN: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',

  devWhitelist: ['http://localhost:3000', 'http://localhost:3001'],
  prodWhitelist: ['https://hongbomiao.com', 'https://www.hongbomiao.com'],
};

export default Config;
