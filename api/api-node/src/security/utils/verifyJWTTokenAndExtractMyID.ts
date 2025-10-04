import jsonwebtoken, { JwtPayload } from 'jsonwebtoken';
import config from '../../config.js';

type DecodedToken = JwtPayload & {
  id: string;
  email: string;
};

const verifyJWTTokenAndExtractMyID = (authorization: string | undefined): string | null => {
  if (authorization == null) {
    return null;
  }
  const token = authorization.replace('Bearer ', '');
  const { id } = jsonwebtoken.verify(token, config.jwtSecret) as DecodedToken;
  return id;
};

export default verifyJWTTokenAndExtractMyID;
