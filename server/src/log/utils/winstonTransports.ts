import fluentLogger from 'fluent-logger';
import winston from 'winston';
import Config from '../../Config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const consoleFormats = isProduction()
  ? [winston.format.timestamp(), winston.format.json()]
  : [winston.format.timestamp(), winston.format.prettyPrint(), winston.format.colorize({ all: true })];

const winstonTransports = [
  ...(Config.shouldShowLog
    ? [
        new winston.transports.Console({
          format: winston.format.combine(...consoleFormats),
        }),
      ]
    : []),

  ...(isDevelopment() ? [new (fluentLogger.support.winstonTransport())('hm-server', Config.fluentBitConfig)] : []),
];

export default winstonTransports;
