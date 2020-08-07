import logger from '../../log/utils/logger';

const cleanup = async (): Promise<void> => {
  logger.info('Server is starting cleanup.');
};

export default cleanup;
