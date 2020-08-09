import { GraphQLObjectType } from 'graphql';
import MeGraphQLType from '../graphQLTypes/Me.type';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    me: {
      type: MeGraphQLType,
      resolve: () => {
        return { name: 'Hongbo Miao', slogan: 'Making magic happen' };
      },
    },
  },
});

export default query;
