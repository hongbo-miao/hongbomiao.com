import { ISitemapOptions, Sitemap } from 'sitemap';

const getSitemap = (options: ISitemapOptions): string => String(new Sitemap(options));

export default getSitemap;
