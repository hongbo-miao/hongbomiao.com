import crypto from 'crypto';


function getScriptSrcHashes(index) {
  let hashes = '';
  const matches = index.match(/<script>.+?<\/script>/g);

  if (matches) {
    matches.forEach((scriptTag) => {
      const hash = crypto.createHash('sha256');
      const content = scriptTag
        .replace('<script>', '')
        .replace('</script>', '');

      const sha256 = hash
        .update(content)
        .digest('base64');

      hashes += `'sha256-${sha256}' `;
    });
  }

  return hashes;
}

export default getScriptSrcHashes;
