import * as Sentry from '@sentry/node';
import Config from '../config';

const initSentry = () => {
  Sentry.init({
    dsn: Config.sentryDSN,
    environment: Config.nodeEnv,
  });
};

export default initSentry;
