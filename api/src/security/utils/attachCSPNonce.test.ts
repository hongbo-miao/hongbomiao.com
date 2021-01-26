import attachCSPNonce from './attachCSPNonce';

describe('attachCSPNonce', () => {
  test('should attach nonce', () => {
    const html = '<!doctype html><script>a</script>b<script>c</script><script src="/static/js/1.js"></script></html>';
    const newHTML = attachCSPNonce(html, 'sh2y6CU26atfrWwfdutNKw==');
    expect(newHTML).toEqual(
      '<!doctype html><script nonce="sh2y6CU26atfrWwfdutNKw==">a</script>b<script nonce="sh2y6CU26atfrWwfdutNKw==">c</script><script nonce="sh2y6CU26atfrWwfdutNKw==" src="/static/js/1.js"></script></html>'
    );
  });
});
