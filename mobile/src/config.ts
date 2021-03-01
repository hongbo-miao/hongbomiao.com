// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
// eslint-disable-next-line import/no-unresolved
import { API_URL } from '@env';

if (API_URL == null || API_URL === '') {
  throw new Error('Failed to read API_URL.');
}

type Config = {
  graphQLURL: string;
};

const config: Config = {
  graphQLURL: `${API_URL}/graphql`,
};

export default config;
