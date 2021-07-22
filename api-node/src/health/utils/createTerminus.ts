import http from 'http';
import terminus from '@godaddy/terminus';
import spdy from 'spdy';
import checkHealth from './checkHealth';
import cleanup from './cleanup';

const createTerminus = (server: http.Server | spdy.Server): void => {
  terminus.createTerminus(server, {
    signals: ['SIGINT', 'SIGBREAK', 'SIGHUP', 'SIGTERM'],
    healthChecks: {
      '/health': checkHealth,
      verbatim: true,
    },
    onSignal: cleanup,
  });
};

export default createTerminus;
