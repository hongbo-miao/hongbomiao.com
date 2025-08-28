import config from '../../config';

const graphQLFetch = async (query: string): Promise<Response> => {
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

  return response;
};

export default graphQLFetch;
