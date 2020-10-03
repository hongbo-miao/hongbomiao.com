import { print } from 'graphql';
import gql from 'graphql-tag';

const meQuery = print(gql`
  query Me {
    me {
      name
      bio
    }
  }
`);

export default meQuery;
