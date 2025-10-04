import Redis from 'ioredis';
import config from '../../config.js';

const redis = new Redis(config.redisOptions);

export default redis;
