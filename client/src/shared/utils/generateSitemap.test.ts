import { EnumChangefreq, SitemapItemLoose, SitemapStreamOptions } from 'sitemap';
import generateSitemap from '../../../scripts/generateSitemap';

describe('generateSitemap', () => {
  test('get sitemap', async () => {
    const options: SitemapStreamOptions = {
      hostname: 'https://example.com',
    };
    const links: ReadonlyArray<SitemapItemLoose> = [{ url: '/', changefreq: EnumChangefreq.HOURLY, priority: 1 }];
    const sitemap = await generateSitemap(options, links);
    expect(sitemap).toMatchSnapshot();
  });
});
