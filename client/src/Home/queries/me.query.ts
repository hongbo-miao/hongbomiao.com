import { print } from 'graphql';
import gql from 'graphql-tag';
import meNamesFragment from './meNames.fragment';

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
