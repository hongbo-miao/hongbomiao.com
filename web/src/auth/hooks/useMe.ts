import axios, { AxiosResponse } from 'axios';
import { useState } from 'react';
import { useQuery, useQueryClient } from 'react-query';
import queryKeys from '../../shared/reactQuery/queryKeys';
import LocalStorageMe from '../types/LocalStorageMe';
import LocalStorage from '../utils/LocalStorage';
import axiosInstance from '../utils/axiosInstance';
import getJWTHeader from '../utils/getJWTHeader';

async function getUser(me: LocalStorageMe | null): Promise<AxiosResponse | null> {
  const source = axios.CancelToken.source();

  if (me == null) return null;
  return axiosInstance({
    headers: getJWTHeader(me),
    cancelToken: source.token,
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
}

interface useMe {
  me: LocalStorageMe | null;
  updateMe: (me: LocalStorageMe) => void;
  clearMe: () => void;
}

const useMe = (): useMe => {
  const [me, setMe] = useState<LocalStorageMe | null>(LocalStorage.getMe());
  const queryClient = useQueryClient();

  useQuery(queryKeys.me, () => getUser(me), {
    enabled: !!me,
    onSuccess: (axiosResponse) => setMe(axiosResponse?.data?.me),
  });

  const updateMe = (newUser: LocalStorageMe): void => {
    setMe(newUser);
    LocalStorage.setMe(newUser);

    queryClient.setQueryData(queryKeys.me, newUser);
  };

  const clearMe = () => {
    setMe(null);
    LocalStorage.clearMe();

    queryClient.setQueryData(queryKeys.me, null);
    queryClient.removeQueries([queryKeys.me]);
  };

  return { me, updateMe, clearMe };
};

export default useMe;
