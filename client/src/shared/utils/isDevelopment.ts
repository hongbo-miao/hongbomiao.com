import Config from '../../Config';
import NodeEnv from './NodeEnv';

const isDevelopment = (nodeEnv = Config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.development;
};

export default isDevelopment;
