import winston, { createLogger } from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../config';

const logger = createLogger({
  transports: [
    new winston.transports.Console({ level: 'error' }),
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    new SentryTransport(Config.sentryOptions),
  ],
});

export default logger;
