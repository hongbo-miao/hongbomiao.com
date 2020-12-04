import { createClient } from 'graphql-ws';
import config from '../../config';

const graphQLSubscriptionClient = createClient({
  url: config.webSocketGraphQLURL,
});

export default graphQLSubscriptionClient;
