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
  githubUrl: string;
  serverApiBaseUrl: string;
  serverWebSocketBaseUrl: string;
  googleTagManagerOptions: {
    containerId: string;
  };
  lightstep: {
    token: string;
    traceUrl: string;
  };
  sentryOptions: Sentry.BrowserOptions;
};

const config: Config = {
  nodeEnv: MODE,
  githubUrl: 'https://github.com/hongbo-miao/hongbomiao.com',
  serverApiBaseUrl: MODE === 'development' ? 'http://localhost:58136' : '',
  serverWebSocketBaseUrl: `${VITE_SERVER_WS_PROTOCOL}://${window.location.host}`,
  googleTagManagerOptions: {
    containerId: 'GTM-MKMQ55P',
  },
  lightstep: {
    token: 'W2sFPG0uCgnCAjr/d0NfngMArOSUEb1SN/5UlOLnZxQ3/4hWndgg/J3jZX74b/c0AF4+o+h0lRGY2vHHFWJBuMh4CKMyILo3pMznB4xd',
    traceUrl: 'https://ingest.lightstep.com/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://a0ff55d9ee00403ca144425a33c318eb@o379185.ingest.sentry.io/4504195581018112',
    environment: MODE,
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: 1.0,
  },
};

export default config;
