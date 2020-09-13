import Config from '../../Config';
import NodeEnv from './NodeEnv';

const isProduction = (nodeEnv = Config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.production;
};

export default isProduction;
