import { ISitemapOptions, Sitemap } from 'sitemap';


function getSitemap(options: ISitemapOptions): string {
  return String(new Sitemap(options));
}

export default getSitemap;
