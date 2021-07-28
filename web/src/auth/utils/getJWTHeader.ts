import Me from '../types/Me';

interface JWTHeader {
  Authorization?: string;
}

const getJWTHeader = (me: Me): JWTHeader => {
  return { Authorization: `Bearer ${me.jwtToken}` };
};

export default getJWTHeader;
