import { GraphQLObjectType } from 'graphql';
import MeType from '../types/me.type';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    me: {
      type: MeType,
      resolve() {
        return { name: 'Hongbo Miao', slogan: 'Making magic happen' };
      },
    },
  },
});

export default query;
