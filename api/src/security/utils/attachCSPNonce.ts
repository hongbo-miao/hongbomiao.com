const attachCSPNonce = (html: string, cspNonce: string): string => {
  return html.replace(/<script/g, `<script nonce="${cspNonce}"`).replace(/<style/g, `<style nonce="${cspNonce}"`);
};

export default attachCSPNonce;
