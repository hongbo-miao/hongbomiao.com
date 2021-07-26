import { AxiosResponse } from 'axios';
import React from 'react';
import { useQuery, useQueryClient } from 'react-query';
import queryKeys from '../../shared/reactQuery/queryKeys';
import LocalStorageMe from '../types/LocalStorageMe';
import LocalStorage from '../utils/LocalStorage';
import axiosInstance from '../utils/axiosInstance';
import getJWTHeader from '../utils/getJWTHeader';

const getMe = async (me: LocalStorageMe | null): Promise<AxiosResponse | null> => {
  if (me == null) return null;
  return axiosInstance({
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

interface useMe {
  me: LocalStorageMe | null;
  updateMe: (me: LocalStorageMe) => void;
  clearMe: () => void;
}

const useMe = (): useMe => {
  const [me, setMe] = React.useState<LocalStorageMe | null>(LocalStorage.getMe());
  const queryClient = useQueryClient();

  useQuery(queryKeys.me, () => getMe(me), {
    enabled: me != null,
    onSuccess: (axiosResponse) => {
      return setMe(axiosResponse?.data?.data?.me);
    },
  });

  const updateMe = (newMe: LocalStorageMe): void => {
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
