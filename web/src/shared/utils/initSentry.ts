import * as Sentry from '@sentry/react';
import config from '../../config';

const initSentry = (): void => {
  Sentry.init(config.sentryOptions);
};

export default initSentry;
