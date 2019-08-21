import path from 'path';
import { promises as fsp } from 'fs';
import { Sitemap } from 'sitemap';

import Paths from './paths';


async function buildSitemap() {
  const sitemap = new Sitemap({
    hostname: 'https://hongbomiao.com',
    cacheTime: 10 * 60 * 1000, // 10 min, cache purge period
    urls: [
      { url: Paths.appRootPath, changefreq: 'hourly', priority: 1 },
    ],
  });

  const sitemapPath = path.resolve(__dirname, '..', '..', '..', 'public', 'sitemap.xml');
  await fsp.writeFile(sitemapPath, String(sitemap));
}

buildSitemap();
