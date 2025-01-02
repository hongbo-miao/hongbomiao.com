import { print, parse } from 'graphql';
import meNamesFragment from './meNamesFragment';

const meQuery = print(
  parse(`
    query Me {
      me {
        name
        bio
        ...meNames
      }
    }
    ${meNamesFragment}
  `),
);

export default meQuery;
