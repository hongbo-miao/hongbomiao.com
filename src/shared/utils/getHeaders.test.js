import getHeaders from './getHeaders';


describe('getHeaders', () => {
  test('insert hash in script-src', () => {
    const headers = `/*
      Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com
      X-Frame-Options: DENY`;
    const hashes = "'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' ";

    const newHeaders = getHeaders(headers, hashes);
    const expectedHeaders = `/*
      Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com
      X-Frame-Options: DENY`;
    expect(newHeaders).toBe(expectedHeaders);
  });

  test('no insert if empty hash', () => {
    const headers = `/*
      Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com
      X-Frame-Options: DENY`;
    const hashes = '';

    const newHeaders = getHeaders(headers, hashes);
    expect(newHeaders).toBe(headers);
  });
});
