import { AxiosResponse } from 'axios';
import LocalStorage from '../../auth/utils/LocalStorage';
import axiosInstance from '../../auth/utils/axiosInstance';
import getJWTHeader from '../../auth/utils/getJWTHeader';
import config from '../../config';

const adoptDog = async (id: string): Promise<AxiosResponse | null> => {
  const localStorageMe = LocalStorage.getMe();
  if (localStorageMe == null) return null;

  return axiosInstance({
    baseURL: config.graphqlServerGraphQLURL,
    headers: getJWTHeader(localStorageMe),
    data: {
      query: `
        mutation AdoptDog($id: ID!) {
          adoptDog(id: $id) {
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

export default adoptDog;
