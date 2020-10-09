type Config = {
  serverURL: string;
  autocannon: {
    connections: number;
    amount: number;
  };
};

const config: Config = {
  serverURL: 'http://localhost:5000',
  autocannon: {
    connections: 5,
    amount: 500,
  },
};

export default config;
