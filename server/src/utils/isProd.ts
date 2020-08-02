import Config from '../config';

const isProd = Config.nodeEnv === 'production';

export default isProd;
