import fluentLogger from 'fluent-logger';
import winston from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const formats = isProduction()
  ? [winston.format.timestamp(), winston.format.json()]
  : [winston.format.timestamp(), winston.format.prettyPrint(), winston.format.colorize({ all: true })];

const winstonTransports = [
  new winston.transports.Console({
    format: winston.format.combine(...formats),
  }),

  ...(isDevelopment() ? [new (fluentLogger.support.winstonTransport())('hm-server', Config.fluentBitConfig)] : []),

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  ...(isDevelopment() || isProduction() ? [new SentryTransport(Config.sentryOptions)] : []),
];

export default winstonTransports;
