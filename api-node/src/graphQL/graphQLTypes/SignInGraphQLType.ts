import { GraphQLObjectType, GraphQLString } from 'graphql';

const SignInGraphQLType = new GraphQLObjectType({
  name: 'SignIn',
  fields: {
    jwtToken: { type: GraphQLString },
  },
});

export default SignInGraphQLType;
