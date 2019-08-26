import getScriptSrcHashes from '../../../bin/getScriptSrcHashes';


describe('getScriptSrcHashes', () => {
  test('hash usage for script element with content', () => {
    const index = '<!doctype html><script>a</script>b<script>c</script><script src="/static/js/1.js"></script></html>';
    const hashes = getScriptSrcHashes(index);
    expect(hashes).toBe("'sha256-ypeBEsobvcr6wjGzmiPcTaeG7/gUfE5yuYB3ha/uSLs=' 'sha256-Ln0sA6lQeuJl7PW1NWiFpTOTogKdJBOUmXJloaJa78Y=' ");
  });

  test('return empty hash if no script element with content', () => {
    const index = '<!doctype html><script src="/static/js/1.js"></script></html>';
    const hashes = getScriptSrcHashes(index);
    expect(hashes).toBe('');
  });
});
