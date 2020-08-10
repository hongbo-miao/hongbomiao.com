import * as Sentry from '@sentry/react';
import Config from '../../Config';

const initSentry = (): void => {
  Sentry.init(Config.sentryOptions);
};

export default initSentry;
