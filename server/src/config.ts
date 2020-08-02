const Config = {
  nodeEnv: process.env.NODE_ENV,
  port: process.env.PORT,

  devWhitelist: ['http://localhost:3000', 'http://localhost:3001'],
  prodWhitelist: ['https://hongbomiao.com', 'https://www.hongbomiao.com'],
};

export default Config;
