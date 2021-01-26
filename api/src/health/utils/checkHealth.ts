type Health = {
  api: string;
};

const checkHealth = async (): Promise<Health> => {
  return Promise.resolve({ api: 'ok' });
};

export default checkHealth;
