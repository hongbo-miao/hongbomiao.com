import fluentLogger from 'fluent-logger';
import winston from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const winstonTransports = [
  new winston.transports.Console({
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.colorize({ all: true }),
      winston.format.printf((info) => {
        const { timestamp, level, message, ...args } = info;
        return `${timestamp} ${level}: ${message} ${Object.keys(args).length ? JSON.stringify(args, null, 2) : ''}`;
      })
    ),
  }),

  ...(isDevelopment() ? [new (fluentLogger.support.winstonTransport())('hm-server', Config.fluentBitConfig)] : []),

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  ...(isDevelopment() || isProduction() ? [new SentryTransport(Config.sentryOptions)] : []),
];

export default winstonTransports;
