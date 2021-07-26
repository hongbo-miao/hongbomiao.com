import LocalStorageMe from '../types/LocalStorageMe';

interface JWTHeader {
  Authorization?: string;
}

const getJWTHeader = (me: LocalStorageMe): JWTHeader => {
  return { Authorization: `Bearer ${me.jwtToken}` };
};

export default getJWTHeader;
