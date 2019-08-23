import path from 'path';
import { promises as fsp } from 'fs';

import insertHashesToScriptSrc from '../src/shared/utils/insertHashesToScriptSrc';
import getScriptSrcHashes from '../src/shared/utils/getScriptSrcHashes';


async function updateHeaders() {
  const headersPath = path.resolve(__dirname, '..', 'build', '_headers');
  const headers = await fsp.readFile(headersPath, 'utf-8');

  const indexPath = path.resolve(__dirname, '..', 'build', 'index.html');
  const index = await fsp.readFile(indexPath, 'utf-8');
  const hashes = getScriptSrcHashes(index);

  const newHeaders = insertHashesToScriptSrc(headers, hashes);
  await fsp.writeFile(headersPath, newHeaders);
}

export default updateHeaders;
