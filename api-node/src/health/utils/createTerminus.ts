import http from 'http';
import terminus from '@godaddy/terminus';
import spdy from 'spdy';
import checkHealth from './checkHealth';

const createTerminus = (server: http.Server | spdy.Server): void => {
  terminus.createTerminus(server, {
    signals: ['SIGINT', 'SIGBREAK', 'SIGHUP', 'SIGTERM'],
    healthChecks: {
      '/health': checkHealth,
      verbatim: true,
    },
  });
};

export default createTerminus;
