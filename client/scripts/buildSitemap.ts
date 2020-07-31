import path from 'path';
import { EnumChangefreq, SitemapItemLoose, SitemapStreamOptions } from 'sitemap';
import { promises as fsp } from 'fs';

import getSitemap from './getSitemap';
import Paths from '../src/shared/utils/paths';

const buildSitemap = async (): Promise<void> => {
  const options: SitemapStreamOptions = {
    hostname: 'https://hongbomiao.com',
  };
  const links: SitemapItemLoose[] = [{ url: Paths.appRootPath, changefreq: EnumChangefreq.HOURLY, priority: 1 }];
  const sitemap = await getSitemap(options, links);
  const sitemapPath = path.resolve(__dirname, '../public/sitemap.xml');
  await fsp.writeFile(sitemapPath, sitemap);
};

export default buildSitemap;
