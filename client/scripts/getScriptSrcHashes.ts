import crypto from 'crypto';

const getScriptSrcHashes = (html: string): string => {
  let hashes = '';
  const matches = html.match(/<script>.+?<\/script>/g);

  if (matches) {
    matches.forEach((scriptTag) => {
      const content = scriptTag.replace('<script>', '').replace('</script>', '');
      const sha256 = crypto.createHash('sha256').update(content).digest('base64');
      hashes += `'sha256-${sha256}' `;
    });
  }

  return hashes;
};

export default getScriptSrcHashes;
