import config from '../../config';
import logger from './logger';

const printStatus = (): void => {
  logger.info(`NODE_ENV: ${config.nodeEnv}`);
  logger.info(`PORT: ${config.port}`);
};

export default printStatus;
