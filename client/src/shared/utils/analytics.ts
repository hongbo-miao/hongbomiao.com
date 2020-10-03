// eslint-disable-next-line @typescript-eslint/ban-ts-ignore
// @ts-ignore
import googleTagManager from '@analytics/google-tag-manager';
import Analytics from 'analytics';
import config from '../../config';
import isDevelopment from './isDevelopment';
import isProduction from './isProduction';

const analytics = Analytics({
  app: 'hm-client-analytics',
  debug: isDevelopment(),
  plugins: isProduction() ? [googleTagManager(config.googleTagManagerOptions)] : [],
});

export default analytics;
