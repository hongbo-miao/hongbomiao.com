import config from '../../config';

const isProduction = config.nodeEnv === 'production';

export default isProduction;
