import { AxiosResponse } from 'axios';
import LocalStorage from '../../auth/utils/LocalStorage';
import axiosInstance from '../../auth/utils/axiosInstance';
import getJWTHeader from '../../auth/utils/getJWTHeader';
import config from '../../config';

const getOPALDecision = async (action: string, resourceType: string): Promise<AxiosResponse | null> => {
  const localStorageMe = LocalStorage.getMe();
  if (localStorageMe == null) return null;

  return axiosInstance({
    baseURL: config.graphqlServerGraphQLURL,
    headers: getJWTHeader(localStorageMe),
    data: {
      query: `
        query OPAL(
          $action: String!
          $resourceType: String!
        ) {
          opal(
            action: $action
            resourceType: $resourceType
          ) {
            decision
          }
        }
      `,
      variables: {
        action,
        resourceType,
      },
    },
  });
};

export default getOPALDecision;
