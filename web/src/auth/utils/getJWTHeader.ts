import LocalStorageMe from '../types/LocalStorageMe';

interface JWTHeader {
  Authorization?: string;
}

const getJWTHeader = (user: LocalStorageMe): JWTHeader => {
  return { Authorization: `Bearer ${user.jwtToken}` };
};

export default getJWTHeader;
