import { useQuery } from '@tanstack/react-query';
import config from '@/config';
import pingSubscription from '@/health/queries/pingSubscription';
import type { GraphQLPing } from '@/health/types/GraphQLPing';
import type { GraphQLResponse } from '@/shared/types/GraphQLResponse';

const fetchPing = async (query: string): Promise<GraphQLResponse<GraphQLPing>> => {
  const response = await fetch(config.graphqlServerGraphQLURL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

const useHealthSubscription = () => {
  return useQuery({
    queryKey: ['health', 'ping'],
    queryFn: () => fetchPing(pingSubscription),
    refetchInterval: 30000, // 30 seconds
  });
};

export default useHealthSubscription;
