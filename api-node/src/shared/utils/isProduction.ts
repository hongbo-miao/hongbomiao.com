import config from '../../config';
import NodeEnv from './NodeEnv';

const isProduction = (nodeEnv = config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.production;
};

export default isProduction;
