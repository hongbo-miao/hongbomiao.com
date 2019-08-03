import path from 'path';
import sm from 'sitemap';
import fs from 'fs';

import Paths from './shared/utils/paths';


const OUTPUT_FILE = path.resolve(__dirname, '..', 'public', 'sitemap.xml');

const sitemap = sm.createSitemap({
  hostname: 'https://hongbomiao.com',
  cacheTime: 10 * 60 * 1000,  // 10 min, cache purge period
  urls: [
    { url: Paths.appRootPath, changefreq: 'hourly', priority: 1 },
  ],
});

fs.writeFileSync(OUTPUT_FILE, sitemap.toString());

console.log(`Sitemap written at ${OUTPUT_FILE}`);
