import path from 'path';
import fs from 'fs';
import { Sitemap } from 'sitemap';

import Paths from './paths';


const OUTPUT_FILE = path.resolve(__dirname, '..', '..', '..', 'public', 'sitemap.xml');

const sitemap = new Sitemap({
  hostname: 'https://hongbomiao.com',
  cacheTime: 10 * 60 * 1000, // 10 min, cache purge period
  urls: [
    { url: Paths.appRootPath, changefreq: 'hourly', priority: 1 },
  ],
});

fs.writeFileSync(OUTPUT_FILE, sitemap.toString());
