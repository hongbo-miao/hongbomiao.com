import config from '../../config';

const isProd = config.nodeEnv === 'production';

export default isProd;
