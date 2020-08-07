import http from 'http';
import terminus from '@godaddy/terminus';
import spdy from 'spdy';
import checkHealth from './checkHealth';
import cleanup from './cleanup';

const createTerminus = (server: http.Server | spdy.Server) => {
  return terminus.createTerminus(server, {
    signal: 'SIGINT',
    healthChecks: {
      verbatim: true,
      '/health': checkHealth,
    },
    onSignal: cleanup,
  });
};

export default createTerminus;
