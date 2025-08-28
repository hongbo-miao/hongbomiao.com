import { createClient } from 'graphql-ws';
import config from '@/config';

const graphqlSubscriptionClient = createClient({
  url: `${config.serverWebSocketBaseUrl}/graphql`,
});

export default graphqlSubscriptionClient;
