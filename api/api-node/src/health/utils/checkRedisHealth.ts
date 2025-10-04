import redis from '../../security/utils/redis.js';

const checkRedisHealth = async (): Promise<boolean> => {
  const pong = await redis.ping();
  return pong === 'PONG';
};

export default checkRedisHealth;
