import * as LogRocket from 'logrocket';
import setupLogRocketReact from 'logrocket-react';
import Config from '../../Config';

const initLogRocket = (): void => {
  LogRocket.init(Config.logRocketAppID);
  setupLogRocketReact(LogRocket);
};

export default initLogRocket;
