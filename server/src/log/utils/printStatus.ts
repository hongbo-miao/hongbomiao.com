import Config from '../../config';
import logger from './logger';

const printStatus = (): void => {
  logger.info(`NODE_ENV: ${Config.nodeEnv}`);
  logger.info(`PORT: ${Config.port}`);
};

export default printStatus;
