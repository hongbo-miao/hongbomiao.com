import config from '@/config';
import NodeEnv from '@/shared/utils/NodeEnv';

const isDevelopment = (nodeEnv = config.nodeEnv): boolean => {
  return nodeEnv === NodeEnv.development;
};

export default isDevelopment;
