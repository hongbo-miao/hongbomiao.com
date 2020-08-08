const { REACT_APP_DOMAIN, REACT_APP_PORT, NODE_ENV } = process.env;

const config = {
  nodeEnv: NODE_ENV,

  githubUrl: 'https://github.com/hongbo-miao/hongbomiao.com',
  graphQLURL: `https://${REACT_APP_DOMAIN}:${REACT_APP_PORT}/graphql`,
};

export default config;
