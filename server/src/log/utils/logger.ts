import pino from 'pino';
import isDevelopment from '../../shared/utils/isDevelopment';

const logger = pino({
  prettyPrint: isDevelopment(),
});

export default logger;
