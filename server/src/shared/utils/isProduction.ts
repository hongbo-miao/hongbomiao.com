import Config from '../../Config';

const isProduction = (nodeEnv = Config.nodeEnv): boolean => {
  return nodeEnv === 'production';
};

export default isProduction;
