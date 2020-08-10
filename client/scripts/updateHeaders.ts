import { promises as fsp } from 'fs';
import path from 'path';
import createScriptSrcHashes from './createScriptSrcHashes';
import insertHashesToScriptSrc from './insertHashesToScriptSrc';

const updateHeaders = async (): Promise<void> => {
  const headersPath = path.join(__dirname, '../build/_headers');
  const headers = await fsp.readFile(headersPath, 'utf-8');

  const indexPath = path.join(__dirname, '../build/index.html');
  const indexHTML = await fsp.readFile(indexPath, 'utf-8');
  const hashes = createScriptSrcHashes(indexHTML);

  const newHeaders = insertHashesToScriptSrc(headers, hashes);
  await fsp.writeFile(headersPath, newHeaders);
};

export default updateHeaders;
