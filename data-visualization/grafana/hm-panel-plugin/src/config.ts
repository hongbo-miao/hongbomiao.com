type Config = {
  httpServerURL: string;
  webSocketServerURL: string;
};

const isProduction = window.location.protocol === 'https:';
const config: Config = {
  httpServerURL: isProduction ? 'https://localhost:35903' : 'http://localhost:35903',
  webSocketServerURL: isProduction ? `wss://localhost:35903` : `ws://localhost:35903`,
};

export default config;
