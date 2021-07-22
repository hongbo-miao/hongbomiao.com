import pino from 'pino';
import logger from '../../log/utils/logger';

const handler = pino.final(logger, (err, finalLogger) => {
  finalLogger.info('Server is starting cleanup.');
});

const cleanup = async (): Promise<void> => {
  handler(null);
};

export default cleanup;
