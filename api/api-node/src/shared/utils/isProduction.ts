import config from '../../config.js';
import NodeEnv from './NodeEnv.js';

const isProduction = (nodeEnv = config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.production;
};

export default isProduction;
