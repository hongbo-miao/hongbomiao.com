import { print } from 'graphql';
import gql from 'graphql-tag';

const pingSubscription = print(gql`
  subscription {
    ping
  }
`);

export default pingSubscription;
