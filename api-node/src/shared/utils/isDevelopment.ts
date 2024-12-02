import config from '../../config.js';
import NodeEnv from './NodeEnv.js';

const isDevelopment = (nodeEnv = config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.development;
};

export default isDevelopment;
