import { print } from 'graphql';
import gql from 'graphql-tag';
import meNamesFragment from './meNamesFragment';

const meQuery = print(gql`
  query Me {
    me {
      name
      bio
      ...meNames
    }
  }
  ${meNamesFragment}
`);

export default meQuery;
