function insertHashesToScriptSrc(headers, hashes) {
  const scriptSrc = `script-src 'self' ${hashes}`;
  return headers.replace('script-src \'self\' ', scriptSrc);
}

export default insertHashesToScriptSrc;
