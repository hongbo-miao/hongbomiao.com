import Config from '../../Config';

const isProduction = Config.nodeEnv === 'production' && Config.domain !== 'localhost';

export default isProduction;
