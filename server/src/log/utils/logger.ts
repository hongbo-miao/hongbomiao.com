import winston from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';
import winstonTransports from './winstonTransports';

const logger = winston.createLogger({
  transports: [
    ...winstonTransports,

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    ...(isDevelopment() || isProduction() ? [new SentryTransport(Config.sentryOptions)] : []),
  ],
});

export default logger;
