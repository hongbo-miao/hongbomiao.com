import { promises as fsp } from 'fs';
import path from 'path';
import { EnumChangefreq, SitemapItemLoose, SitemapStreamOptions } from 'sitemap';
import Paths from '../src/shared/utils/paths';
import generateSitemap from './generateSitemap';

const buildSitemap = async (): Promise<void> => {
  const options: SitemapStreamOptions = {
    hostname: 'https://www.hongbomiao.com',
  };
  const links: SitemapItemLoose[] = [{ url: Paths.appRootPath, changefreq: EnumChangefreq.HOURLY, priority: 1 }];
  const sitemap = await generateSitemap(options, links);
  await fsp.writeFile(path.join(__dirname, '../public/sitemap.xml'), sitemap);
};

export default buildSitemap;
