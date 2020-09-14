import fluentLogger from 'fluent-logger';
import winston, { createLogger } from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const logger = createLogger();

logger.add(
  new winston.transports.Console({
    format: winston.format.combine(winston.format.colorize({ all: true })),
  })
);

if (isDevelopment()) {
  const FluentTransport = fluentLogger.support.winstonTransport();
  logger.add(new FluentTransport('hm-server', Config.fluentBitConfig));
}

if (isDevelopment() || isProduction()) {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  logger.add(new SentryTransport(Config.sentryOptions));
}

export default logger;
