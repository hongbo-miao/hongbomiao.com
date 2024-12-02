import { promises as fsp } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { EnumChangefreq, SitemapItemLoose, SitemapStreamOptions } from 'sitemap';
import Paths from '../src/shared/utils/paths.js';
import generateSitemap from './generateSitemap.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const buildSitemap = async (): Promise<void> => {
  const options: SitemapStreamOptions = {
    hostname: 'https://www.hongbomiao.com',
  };
  const links: ReadonlyArray<SitemapItemLoose> = [
    { url: Paths.appRootPath, changefreq: EnumChangefreq.HOURLY, priority: 1 },
  ];
  const sitemap = await generateSitemap(options, links);
  await fsp.writeFile(path.join(__dirname, '../public/sitemap.xml'), sitemap);
};

export default buildSitemap;
