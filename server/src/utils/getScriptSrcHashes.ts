import crypto from 'crypto';

const getScriptSrcHashes = (index: string): string[] => {
  const hashes: string[] = [];
  const matches = index.match(/<script>.+?<\/script>/g);

  if (matches) {
    matches.forEach((scriptTag) => {
      const content = scriptTag.replace('<script>', '').replace('</script>', '');
      const sha256 = crypto.createHash('sha256').update(content).digest('base64');
      hashes.push(`'sha256-${sha256}'`);
    });
  }

  return hashes;
};

export default getScriptSrcHashes;
