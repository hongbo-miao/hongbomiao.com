import redis from '../../security/utils/redis';

const checkRedisHealth = async (): Promise<boolean> => {
  const pong = await redis.ping();
  return pong === 'PONG';
};

export default checkRedisHealth;
