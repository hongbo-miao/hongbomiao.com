import path from 'path';
import { promises as fsp } from 'fs';
import { Sitemap } from 'sitemap';

import Paths from './paths';


export function getSitemap(hostname, cacheTime, urls) {
  return new Sitemap({
    hostname,
    cacheTime,
    urls,
  });
}

async function buildSitemap() {
  const hostname = 'https://hongbomiao.com';
  const cacheTime = 10 * 60 * 1000; // 10 min, cache purge period
  const urls = [
    { url: Paths.appRootPath, changefreq: 'hourly', priority: 1 },
  ];
  const sitemapPath = path.resolve(__dirname, '..', '..', '..', 'public', 'sitemap.xml');

  const sitemap = getSitemap(hostname, cacheTime, urls);
  await fsp.writeFile(sitemapPath, String(sitemap));
}

export default buildSitemap;
