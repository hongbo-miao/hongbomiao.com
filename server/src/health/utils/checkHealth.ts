type Health = {
  server: string;
};

const checkHealth = async (): Promise<Health> => {
  return Promise.resolve({ server: 'ok' });
};

export default checkHealth;
