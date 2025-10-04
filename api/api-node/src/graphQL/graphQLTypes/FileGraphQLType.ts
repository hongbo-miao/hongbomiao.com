import { GraphQLObjectType, GraphQLString } from 'graphql';

const FileGraphQLType = new GraphQLObjectType({
  name: 'File',
  fields: {
    name: { type: GraphQLString },
  },
});

export default FileGraphQLType;
