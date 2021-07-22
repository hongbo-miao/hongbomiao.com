import jsonwebtoken, { JwtPayload } from 'jsonwebtoken';
import config from '../../config';

type DecodedToken = JwtPayload & {
  id: string;
  email: string;
};

const verifyJWTToken = (authorization: string | undefined): string | null => {
  if (authorization != null) {
    const token = authorization.replace('Bearer ', '');
    const { id } = jsonwebtoken.verify(token, config.jwtSecret) as DecodedToken;
    return id;
  }
  return null;
};

export default verifyJWTToken;
