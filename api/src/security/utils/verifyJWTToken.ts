import jsonwebtoken from 'jsonwebtoken';
import config from '../../config';

type DecodedToken = {
  id: string;
  email: string;
  iat: string;
  exp: string;
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
