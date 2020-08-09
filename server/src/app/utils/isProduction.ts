import Config from '../../Config';

const isProduction = Config.nodeEnv === 'production';

export default isProduction;
