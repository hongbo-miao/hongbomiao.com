import winston, { createLogger } from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import config from '../../config';

const logger = createLogger({
  transports: [
    new winston.transports.Console({ level: 'info' }),
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    new SentryTransport(config.sentryOptions),
  ],
});

export default logger;
