import { AxiosResponse } from 'axios';
import React, { Dispatch, SetStateAction } from 'react';
import { useQuery } from 'react-query';
import LocalStorage from '../../auth/utils/LocalStorage';
import axiosInstance from '../../auth/utils/axiosInstance';
import getJWTHeader from '../../auth/utils/getJWTHeader';

type Props = {
  children: JSX.Element;
  action: string;
  resourceType: string;
  setIsDisabled: Dispatch<SetStateAction<boolean>>;
};

const getPreAuthDecision = async (action: string, resourceType: string): Promise<AxiosResponse | null> => {
  const localStorageMe = LocalStorage.getMe();
  if (localStorageMe == null) return null;

  return axiosInstance({
    headers: getJWTHeader(localStorageMe),
    data: {
      query: `
        query OPA(
          $action: String!
          $resourceType: String!
        ) {
          opa(
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

const PreAuth: React.VFC<Props> = (props) => {
  const { children, action, resourceType, setIsDisabled } = props;
  const query = useQuery([action, resourceType], () => getPreAuthDecision(action, resourceType));

  React.useEffect(() => {
    setIsDisabled(query?.data?.data?.data?.opa?.decision);
  }, [query]);

  return <>{children}</>;
};

export default PreAuth;
