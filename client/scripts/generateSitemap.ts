import { SitemapItemLoose, SitemapStream, SitemapStreamOptions, streamToPromise } from 'sitemap';

const generateSitemap = async (
  options: SitemapStreamOptions,
  links: ReadonlyArray<SitemapItemLoose>
): Promise<string> => {
  const stream = new SitemapStream(options);
  links.forEach((link) => stream.write(link));
  stream.end();
  return streamToPromise(stream).then((data) => String(data));
};

export default generateSitemap;
