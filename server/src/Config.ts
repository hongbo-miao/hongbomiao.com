const { DOMAIN, LIGHTSTEP_TOEKN, NODE_ENV, PORT } = process.env;
const sharedCSPConnectSrc = [
  'https://collector.lightstep.com', // Lightstep
];
const sharedCORSAllowList = [
  'electron://altair', // Altair GraphQL Client
  'null', // Safari reports CSP violation
];

const Config = {
  nodeEnv: NODE_ENV,
  domain: DOMAIN,
  port: PORT,

  lightstepToken: LIGHTSTEP_TOEKN,

  devCSPConnectSrc: [...sharedCSPConnectSrc],
  prodCSPConnectSrc: [...sharedCSPConnectSrc],

  devCORSAllowList: [
    ...sharedCORSAllowList,
    'https://localhost:5000',
    'https://localhost:5001',
    'https://localhost:8080',
  ],
  prodCORSAllowList: [
    ...sharedCORSAllowList,
    'https://hongbomiao.com',
    'https://hongbomiao.herokuapp.com',
    'https://www.hongbomiao.com',
  ],

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default Config;
