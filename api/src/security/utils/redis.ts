import Redis from 'ioredis';
import config from '../../config';

const redis = new Redis(config.redisOptions);

export default redis;
