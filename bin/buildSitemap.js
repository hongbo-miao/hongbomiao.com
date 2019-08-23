import path from 'path';
import { promises as fsp } from 'fs';

import Paths from '../src/shared/utils/paths';
import getSitemap from '../src/shared/utils/getSitemap';


async function buildSitemap() {
  const hostname = 'https://hongbomiao.com';
  const cacheTime = 10 * 60 * 1000; // 10 min, cache purge period
  const urls = [
    { url: Paths.appRootPath, changefreq: 'hourly', priority: 1 },
  ];
  const sitemapPath = path.resolve(__dirname, '..', 'public', 'sitemap.xml');

  const sitemap = getSitemap(hostname, cacheTime, urls);
  await fsp.writeFile(sitemapPath, String(sitemap));
}

export default buildSitemap;
