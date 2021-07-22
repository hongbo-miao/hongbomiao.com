import crypto from 'crypto';

const createCSPNonce = (): string => {
  return crypto.randomBytes(16).toString('base64');
};

export default createCSPNonce;
