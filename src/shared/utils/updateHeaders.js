import crypto from 'crypto';
import path from 'path';
import { promises as fsp } from 'fs';


export function getScriptSrcHashes(index) {
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

export function getHeaders(headers, hashes) {
  const scriptSrc = `script-src 'self' ${hashes}`;
  return headers.replace('script-src \'self\' ', scriptSrc);
}

async function updateHeaders() {
  const headersPath = path.resolve(__dirname, '..', '..', '..', 'build', '_headers');
  const headers = await fsp.readFile(headersPath, 'utf-8');

  const indexPath = path.resolve(__dirname, '..', '..', '..', 'build', 'index.html');
  const index = await fsp.readFile(indexPath, 'utf-8');
  const hashes = getScriptSrcHashes(index);

  const newHeaders = getHeaders(headers, hashes);
  await fsp.writeFile(headersPath, newHeaders);
}

export default updateHeaders;
