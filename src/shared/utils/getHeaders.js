function getHeaders(headers, hashes) {
  const scriptSrc = `script-src 'self' ${hashes}`;
  return headers.replace('script-src \'self\' ', scriptSrc);
}

export default getHeaders;
