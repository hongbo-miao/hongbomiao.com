import gql from 'graphql-tag';

const meNamesFragment = gql`
  fragment meNames on Me {
    firstName
    lastName
  }
`;

export default meNamesFragment;
