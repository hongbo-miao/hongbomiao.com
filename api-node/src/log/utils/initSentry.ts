import * as Sentry from '@sentry/node';
import config from '../../config.js';

const initSentry = (): void => {
  Sentry.init(config.sentryOptions);
};

export default initSentry;
