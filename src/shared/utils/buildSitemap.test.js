import { getSitemap } from './buildSitemap';


describe('buildSitemap', () => {
  test('getSitemap', () => {
    const hostname = 'https://example.com';
    const cacheTime = 10 * 60 * 1000;
    const urls = [
      { url: '/', changefreq: 'hourly', priority: 1 },
    ];
    const sitemap = getSitemap(hostname, cacheTime, urls);
    expect(String(sitemap)).toMatchSnapshot();
  });
});
