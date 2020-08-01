import insertHashesToScriptSrc from '../../../scripts/insertHashesToScriptSrc';

describe('insertHashesToScriptSrc', () => {
  test('insert hash in script-src', () => {
    const headers =
      "Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com";
    const hashes = "'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' ";

    const newHeaders = insertHashesToScriptSrc(headers, hashes);
    const expectedHeaders =
      "Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com";
    expect(newHeaders).toBe(expectedHeaders);
  });

  test('no insert if empty hash', () => {
    const headers =
      "Content-Security-Policy: default-src 'none'; script-src 'self' 'sha256-4oflJWBkAb5jB164BW1XwFrQFIiihu9EmgkYuhXJlUc=' https://example.com; style-src 'self' https://example.com";
    const hashes = '';

    const newHeaders = insertHashesToScriptSrc(headers, hashes);
    expect(newHeaders).toBe(headers);
  });
});
