import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AxiosResponse } from 'axios';
import React from 'react';
import config from '../config';
import Seed from '../types/Seed';
import axiosInstance from '../utils/axiosInstance';
import queryKeys from '../reactQuery/queryKeys';

interface UseSeed {
  seed: Seed;
  updateSeed: (seed: Seed) => void;
}

const getSeed = async (): Promise<AxiosResponse<{ data: { seed: Seed } }>> => {
  return axiosInstance({
    baseURL: config.httpServerURL,
    url: '/seed',
  });
};

const initialSeed: Seed = {
  seedNumber: -1,
};

const useSeed = (): UseSeed => {
  const [seed, setSeed] = React.useState<Seed>(initialSeed);
  const queryClient = useQueryClient();

  useQuery([queryKeys.seed], () => getSeed(), {
    onSuccess: (axiosResponse) => {
      const newSeed = { ...seed, ...axiosResponse?.data };
      return setSeed(newSeed);
    },
  });

  const updateSeed = (deltaSeed: Seed): void => {
    const newSeed = { ...seed, ...deltaSeed };
    setSeed(newSeed);
    queryClient.setQueryData([queryKeys.seed], newSeed);
  };

  return { seed, updateSeed };
};

export default useSeed;
