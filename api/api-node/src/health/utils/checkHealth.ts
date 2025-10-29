import logger from '../../log/utils/logger.js';
import checkAPIHealth from './checkAPIHealth.js';
import checkPostgresHealth from './checkPostgresHealth.js';
import checkRedisHealth from './checkRedisHealth.js';

type HealthStatus = 'ok' | 'error';
type Health = Record<string, HealthStatus>;

const checkHealth = async (): Promise<Health> => {
  const services = [
    { name: 'api', check: checkAPIHealth },
    { name: 'postgres', check: checkPostgresHealth },
    { name: 'redis', check: checkRedisHealth },
  ];
  const results = await Promise.allSettled(services.map((service) => service.check()));
  let health = {};

  results.forEach((res, idx) => {
    let healthStatus: HealthStatus = 'error';
    if (res.status === 'fulfilled' && res.value) {
      healthStatus = 'ok';
    } else {
      logger.error({ name: services[idx].name, res }, 'health');
    }
    health = {
      ...health,
      [services[idx].name]: healthStatus,
    };
  });

  return Promise.resolve(<Health>health);
};

export default checkHealth;
