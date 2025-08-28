type Config = {
  serverUrl: string;
  autocannon: {
    connections: number;
    amount: number;
  };
};

const config: Config = {
  serverUrl: 'http://localhost:58136',
  autocannon: {
    connections: 5,
    amount: 500,
  },
};

export default config;
