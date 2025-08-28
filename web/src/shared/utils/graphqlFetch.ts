import config from '@/config';

const graphqlFetch = async (query: string): Promise<Response> => {
  const response = await fetch(`${config.serverApiBaseUrl}/graphql`, {
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

  return response;
};

export default graphqlFetch;
