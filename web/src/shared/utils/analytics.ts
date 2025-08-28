// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import googleTagManager from '@analytics/google-tag-manager';
import { Analytics } from 'analytics';
import config from '@/config';
import isDevelopment from '@/shared/utils/isDevelopment';
import isProduction from '@/shared/utils/isProduction';

const analytics = Analytics({
  app: 'hm-web-analytics',
  debug: isDevelopment(),
  plugins: isProduction() ? [googleTagManager(config.googleTagManagerOptions)] : [],
});

export default analytics;
