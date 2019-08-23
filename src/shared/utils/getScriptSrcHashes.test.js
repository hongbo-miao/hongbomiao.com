import getScriptSrcHashes from './getScriptSrcHashes';


describe('getScriptSrcHashes', () => {
  test('hash usage for script element with content', () => {
    const index = '<!doctype html><script>a</script>b<script>c</script><script src="/static/js/1.js"></script></html>';
    const hashes = getScriptSrcHashes(index);
    const expectedHashes = "'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' 'sha256-Ln0sA6lQeuJl7PW1NWiFpTOTogKdJBOUmXJloaJa78Y=' ";
    expect(hashes).toBe(expectedHashes);
  });

  test('return empty hash if no script element with content', () => {
    const index = '<!doctype html><script src="/static/js/1.js"></script></html>';
    const hashes = getScriptSrcHashes(index);
    const expectedHashes = '';
    expect(hashes).toBe(expectedHashes);
  });
});
