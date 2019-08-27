const insertHashesToScriptSrc = (headers: string, hashes: string): string => {
  const scriptSrc = `script-src 'self' ${hashes}`;
  return headers.replace("script-src 'self' ", scriptSrc);
};

export default insertHashesToScriptSrc;
