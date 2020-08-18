import Config from '../../Config';

const isDevelopment = (nodeEnv = Config.nodeEnv): boolean => {
  return nodeEnv === 'development';
};

export default isDevelopment;
