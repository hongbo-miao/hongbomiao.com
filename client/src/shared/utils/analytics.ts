// eslint-disable-next-line @typescript-eslint/ban-ts-ignore
// @ts-ignore
import googleTagManager from '@analytics/google-tag-manager';
import Analytics from 'analytics';

const analytics = Analytics({
  app: 'hm-client-analytics',
  plugins: [
    googleTagManager({
      containerId: 'GTM-MKMQ55P',
    }),
  ],
});

export default analytics;
