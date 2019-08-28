import path from 'path';
import { promises as fsp } from 'fs';
import { EnumChangefreq, ISitemapOptions } from 'sitemap';

import Paths from '../src/shared/utils/paths';
import getSitemap from './getSitemap';

const buildSitemap = async (): Promise<void> => {
  const sitemapOptions: ISitemapOptions = {
    hostname: 'https://hongbomiao.com',
    cacheTime: 10 * 60 * 1000, // 10 min, cache purge period
    urls: [{ url: Paths.appRootPath, changefreq: EnumChangefreq.HOURLY, priority: 1 }],
  };
  const sitemapPath = path.resolve(__dirname, '..', 'public', 'sitemap.xml');

  const sitemap = getSitemap(sitemapOptions);
  await fsp.writeFile(sitemapPath, sitemap);
};

export default buildSitemap;
