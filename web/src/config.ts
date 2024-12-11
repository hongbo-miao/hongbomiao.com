import * as Sentry from '@sentry/react';

const { MODE, VITE_SERVER_WS_PROTOCOL } = import.meta.env;

if (MODE !== 'development' && MODE !== 'production' && MODE !== 'test') {
  throw new Error('Failed to read MODE.');
}
if (VITE_SERVER_WS_PROTOCOL == null || VITE_SERVER_WS_PROTOCOL === '') {
  throw new Error('Failed to read VITE_SERVER_WS_PROTOCOL.');
}

type Config = {
  nodeEnv: 'development' | 'production' | 'test';
  githubURL: string;
  graphqlServerGraphQLURL: string;
  webSocketGraphQLURL: string;
  googleTagManagerOptions: {
    containerId: string;
  };
  lightstep: {
    token: string;
    traceURL: string;
  };
  sentryOptions: Sentry.BrowserOptions;
};

const config: Config = {
  nodeEnv: MODE,
  githubURL: 'https://github.com/hongbo-miao/hongbomiao.com',
  graphqlServerGraphQLURL: MODE === 'development' ? 'http://localhost:58136/graphql' : '/graphql',
  webSocketGraphQLURL: `${VITE_SERVER_WS_PROTOCOL}://${window.location.host}/graphql`,
  googleTagManagerOptions: {
    containerId: 'GTM-MKMQ55P',
  },
  lightstep: {
    token: 'W2sFPG0uCgnCAjr/d0NfngMArOSUEb1SN/5UlOLnZxQ3/4hWndgg/J3jZX74b/c0AF4+o+h0lRGY2vHHFWJBuMh4CKMyILo3pMznB4xd',
    traceURL: 'https://ingest.lightstep.com/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://a0ff55d9ee00403ca144425a33c318eb@o379185.ingest.sentry.io/4504195581018112',
    environment: MODE,
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: 1.0,
  },
};

export default config;
