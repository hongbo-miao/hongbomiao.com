import pino from 'pino';
import Config from '../../Config';

const logger = pino({
  prettyPrint: Config.shouldPrettifyLog,
});

export default logger;
