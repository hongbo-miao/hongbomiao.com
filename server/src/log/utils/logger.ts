import pino from 'pino';
import Config from '../../Config';

const logger = pino({
  prettyPrint: Config.shouldPrettifyLog,

  // https://getpino.io/#/docs/redaction
  redact: ['req.body.variables.password'],
});

export default logger;
