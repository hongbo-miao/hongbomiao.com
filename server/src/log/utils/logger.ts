import pino from 'pino';
import config from '../../config';

const logger = pino({
  prettyPrint: config.shouldPrettifyLog,

  // https://getpino.io/#/docs/redaction
  redact: ['req.body.variables.password'],
});

export default logger;
