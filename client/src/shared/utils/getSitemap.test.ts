import { EnumChangefreq, SitemapItemLoose, SitemapStreamOptions } from 'sitemap';
import getSitemap from '../../../scripts/getSitemap';

describe('getSitemap', () => {
  test('get sitemap', async () => {
    const options: SitemapStreamOptions = {
      hostname: 'https://example.com',
    };
    const links: SitemapItemLoose[] = [{ url: '/', changefreq: EnumChangefreq.HOURLY, priority: 1 }];
    const sitemap = await getSitemap(options, links);
    expect(sitemap).toMatchSnapshot();
  });
});
