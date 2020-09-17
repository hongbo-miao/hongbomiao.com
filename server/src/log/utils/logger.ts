import fluentLogger from 'fluent-logger';
import winston, { createLogger } from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const logger = createLogger({
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(winston.format.colorize({ all: true })),
    }),

    ...(isDevelopment() ? [new (fluentLogger.support.winstonTransport())('hm-server', Config.fluentBitConfig)] : []),

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    ...(isDevelopment() || isProduction() ? [new SentryTransport(Config.sentryOptions)] : []),
  ],
});

export default logger;
