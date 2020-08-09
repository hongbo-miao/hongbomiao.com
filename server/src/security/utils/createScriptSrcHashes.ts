import crypto from 'crypto';

const createScriptSrcHashes = (html: string): string[] => {
  if (html == null) return [];

  const hashes: string[] = [];
  const matches = html.match(/<script>.+?<\/script>/g);

  if (matches) {
    matches.forEach((scriptTag) => {
      const content = scriptTag.replace('<script>', '').replace('</script>', '');
      const sha256 = crypto.createHash('sha256').update(content).digest('base64');
      hashes.push(`'sha256-${sha256}'`);
    });
  }

  return hashes;
};

export default createScriptSrcHashes;
