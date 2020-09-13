import fluentLogger from 'fluent-logger';
import winston, { createLogger } from 'winston';
import SentryTransport from 'winston-transport-sentry-node';
import Config from '../../Config';

const FluentTransport = fluentLogger.support.winstonTransport();

const logger = createLogger({
  transports: [
    new winston.transports.Console({ level: 'info' }),
    new FluentTransport('server-tag', Config.fluentBitConfig),
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    new SentryTransport(Config.sentryOptions),
  ],
});

export default logger;
