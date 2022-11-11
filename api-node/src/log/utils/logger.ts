import pino from 'pino';
import config from '../../config';

const logger = pino(
  {
    ...(config.shouldPrettifyLog && { target: 'pino-pretty' }),

    // https://getpino.io/#/docs/redaction
    redact: ['buffer'],
  },
  pino.destination({
    minLength: 4096, // Bytes. Buffer before writing
    sync: false,
  }),
);

/*
 * Asynchronously flush periodically to keep the buffer empty in periods of low activity
 * https://github.com/pinojs/pino/blob/master/docs/asynchronous.md#log-loss-prevention
 */
setInterval(() => {
  logger.flush();
}, 10e3).unref(); // 10s

export default logger;
