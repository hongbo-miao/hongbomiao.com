import { EnumChangefreq, ISitemapOptions } from 'sitemap';

import getSitemap from '../../../bin/getSitemap';

describe('getSitemap', () => {
  test('get sitemap', () => {
    const sitemapOptions: ISitemapOptions = {
      hostname: 'https://example.com',
      cacheTime: 10 * 60 * 1000,
      urls: [{ url: '/', changefreq: EnumChangefreq.HOURLY, priority: 1 }],
    };

    const sitemap = getSitemap(sitemapOptions);
    expect(sitemap).toMatchSnapshot();
  });
});
