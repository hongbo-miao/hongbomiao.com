import { AxiosResponse } from 'axios';
import React from 'react';
import { useQuery, useQueryClient } from 'react-query';
import config from '../../config';
import queryKeys from '../../shared/reactQuery/queryKeys';
import Me from '../types/Me';
import LocalStorage from '../utils/LocalStorage';
import axiosInstance from '../utils/axiosInstance';
import getJWTHeader from '../utils/getJWTHeader';

const getMe = async (me: Me | null): Promise<AxiosResponse | null> => {
  if (me == null) return null;
  return axiosInstance({
    baseURL: config.apiServerGraphQLURL,
    headers: getJWTHeader(me),
    data: {
      query: `
        query Me {
          me {
            name
          }
        }
      `,
    },
  });
};

interface UseMe {
  me: Me | null;
  updateMe: (me: Me) => void;
  clearMe: () => void;
}

const useMe = (): UseMe => {
  const [me, setMe] = React.useState<Me | null>(LocalStorage.getMe());
  const queryClient = useQueryClient();

  useQuery(queryKeys.me, () => getMe(me), {
    enabled: me != null,
    onSuccess: (axiosResponse) => {
      const newMe = { ...me, ...axiosResponse?.data?.data?.me };
      return setMe(newMe);
    },
  });

  const updateMe = (deltaMe: Me): void => {
    const newMe = { ...me, ...deltaMe };
    setMe(newMe);
    LocalStorage.setMe(newMe);
    queryClient.setQueryData(queryKeys.me, newMe);
  };

  const clearMe = () => {
    setMe(null);
    LocalStorage.clearMe();
    queryClient.setQueryData(queryKeys.me, null);
    queryClient.removeQueries(queryKeys.me);
  };

  return { me, updateMe, clearMe };
};

export default useMe;
