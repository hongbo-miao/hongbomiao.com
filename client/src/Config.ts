const { REACT_APP_SERVER_DOMAIN, REACT_APP_LIGHTSTEP_TOEKN, REACT_APP_SERVER_PORT, NODE_ENV } = process.env;

const Config = {
  nodeEnv: NODE_ENV,

  githubURL: 'https://github.com/hongbo-miao/hongbomiao.com',
  graphQLURL: `https://${REACT_APP_SERVER_DOMAIN}:${REACT_APP_SERVER_PORT}/graphql`,

  lightstepToken: REACT_APP_LIGHTSTEP_TOEKN,

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default Config;
