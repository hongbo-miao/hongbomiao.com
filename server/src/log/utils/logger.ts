import pino from 'pino';
import config from '../../config';

const logger = pino(
  {
    prettyPrint: config.shouldPrettifyLog,

    // https://getpino.io/#/docs/redaction
    redact: ['req.body.variables.password'],
  },
  pino.destination({
    minLength: 4096, // Bytes. Buffer before writing
    sync: false,
  })
);

/*
 * Asynchronously flush every 10 seconds to keep the buffer empty in periods of low activity
 * https://github.com/pinojs/pino/blob/master/docs/asynchronous.md#log-loss-prevention
 */
setInterval(() => {
  logger.flush();
}, 10 * 1000).unref();

export default logger;
