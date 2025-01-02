import { parse } from 'graphql';

const meNamesFragment = parse(`
  fragment meNames on Me {
    firstName
    lastName
  }
`);

export default meNamesFragment;
