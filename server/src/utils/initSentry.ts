import * as Sentry from '@sentry/node';
import Config from '../config';

const initSentry = (): void => {
  Sentry.init(Config.sentryOptions);
};

export default initSentry;
