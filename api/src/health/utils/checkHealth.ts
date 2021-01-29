import checkAPIHealth from './checkAPIHealth';
import checkPostgresHealth from './checkPostgresHealth';

type HealthName = 'api' | 'postgres';
type HealthStatus = 'ok' | 'error';
type Health = Record<HealthName, HealthStatus>;

const checkHealth = async (): Promise<Health> => {
  const services = [
    { name: 'api', check: checkAPIHealth },
    { name: 'postgres', check: checkPostgresHealth },
  ];

  const results = await Promise.allSettled(services.map((service) => service.check()));

  let health = {};
  results.forEach((res, idx) => {
    health = {
      ...health,
      // eslint-disable-next-line security/detect-object-injection
      [services[idx].name]: res.status === 'fulfilled' ? 'ok' : 'error',
    };
  });

  return Promise.resolve(<Health>health);
};

export default checkHealth;
