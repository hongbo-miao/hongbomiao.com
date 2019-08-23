import { Sitemap } from 'sitemap';


function getSitemap(hostname, cacheTime, urls) {
  return new Sitemap({
    hostname,
    cacheTime,
    urls,
  });
}

export default getSitemap;
