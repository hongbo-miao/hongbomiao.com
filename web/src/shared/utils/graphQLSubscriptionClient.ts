import { createClient } from 'graphql-ws';
import config from '../../config';
import isProduction from './isProduction';

const graphQLSubscriptionClient = createClient({
  url: isProduction() ? config.prodWebSocketGraphQLURL : config.devWebSocketGraphQLURL,
});

export default graphQLSubscriptionClient;
