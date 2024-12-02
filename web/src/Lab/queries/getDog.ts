import type { AxiosResponse } from 'axios';
import LocalStorage from '../../auth/utils/LocalStorage';
import axiosInstance from '../../auth/utils/axiosInstance';
import getAuthHeaders from '../../auth/utils/getAuthHeaders';
import config from '../../config';
import Dog from '../types/Dog';

const getDog = async (id: string): Promise<AxiosResponse<{ data: { dog: Dog } }> | null> => {
  const localStorageMe = LocalStorage.getMe();
  if (localStorageMe == null) return null;

  return axiosInstance({
    baseURL: config.graphqlServerGraphQLURL,
    headers: getAuthHeaders(localStorageMe),
    data: {
      query: `
        query Dog($id: ID!) {
          dog(id: $id) {
            id
            name
          }
        }
      `,
      variables: {
        id,
      },
    },
  });
};

export default getDog;
