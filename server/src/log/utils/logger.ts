import winston from 'winston';
import winstonTransports from './winstonTransports';

const logger = winston.createLogger({
  transports: winstonTransports,
});

export default logger;
