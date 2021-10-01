import Me from '../types/Me';

const getAuthHeaders = (me: Me): Record<string, string> => {
  return { Authorization: `Bearer ${me.jwtToken}` };
};

export default getAuthHeaders;
