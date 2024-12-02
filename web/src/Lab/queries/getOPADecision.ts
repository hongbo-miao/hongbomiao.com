import type { AxiosResponse } from 'axios';
import LocalStorage from '../../auth/utils/LocalStorage';
import axiosInstance from '../../auth/utils/axiosInstance';
import getAuthHeaders from '../../auth/utils/getAuthHeaders';
import config from '../../config';

const getOPADecision = async (action: string, resource: string): Promise<AxiosResponse | null> => {
  const localStorageMe = LocalStorage.getMe();
  if (localStorageMe == null) return null;

  return axiosInstance({
    baseURL: config.graphqlServerGraphQLURL,
    headers: getAuthHeaders(localStorageMe),
    data: {
      query: `
        query OPA(
          $action: String!
          $resource: String!
        ) {
          opa(
            action: $action
            resource: $resource
          ) {
            decision
          }
        }
      `,
      variables: {
        action,
        resource,
      },
    },
  });
};

export default getOPADecision;
