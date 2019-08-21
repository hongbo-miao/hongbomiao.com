import crypto from 'crypto';
import fs from 'fs';
import path from 'path';


const indexPath = path.resolve(__dirname, '..', '..', '..', 'build', 'index.html');
const index = fs.readFileSync(indexPath, 'utf-8');

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

const headersPath = path.resolve(__dirname, '..', '..', '..', 'build', '_headers');
const headers = fs.readFileSync(headersPath, 'utf-8');

const scriptSrc = `script-src 'self' ${hashes}`;
const newHeaders = headers.replace('script-src \'self\' ', scriptSrc);

fs.writeFileSync(headersPath, newHeaders, 'utf-8');
