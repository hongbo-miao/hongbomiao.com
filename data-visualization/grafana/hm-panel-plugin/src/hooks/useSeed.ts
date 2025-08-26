import { useQuery, useQueryClient } from '@tanstack/react-query';
import config from '../config';
import Seed from '../types/Seed';
import queryKeys from '../reactQuery/queryKeys';

interface UseSeed {
  seed: Seed;
  updateSeed: (seed: Seed) => void;
}

const fetchSeed = async (): Promise<Seed> => {
  const response = await fetch(`${config.httpServerURL}/seed`, {
    method: 'GET',
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch seed: ${response.status} ${response.statusText}`);
  }
  const body: { seed: Seed } = await response.json();
  return body.seed;
};

const initialSeed: Seed = {
  seedNumber: -1,
};

const useSeed = (): UseSeed => {
  const queryClient = useQueryClient();

  const { data } = useQuery<Seed>({
    queryKey: [queryKeys.seed],
    queryFn: fetchSeed,
    placeholderData: initialSeed,
  });

  const seed = data ?? initialSeed;

  const updateSeed = (deltaSeed: Seed): void => {
    const newSeed = { ...seed, ...deltaSeed };
    queryClient.setQueryData([queryKeys.seed], newSeed);
  };

  return { seed, updateSeed };
};

export default useSeed;
