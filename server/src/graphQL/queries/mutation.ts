import { GraphQLID, GraphQLNonNull, GraphQLObjectType, GraphQLString } from 'graphql';
import { GraphQLUpload } from 'graphql-upload';
import formatUser from '../../database/postgres/utils/formatUser';
import updateName from '../../database/postgres/utils/updateName';
import getJWTToken from '../../security/utils/getJWTToken';
import FileGraphQLType from '../graphQLTypes/File.graphQLType';
import SignInGraphQLType from '../graphQLTypes/SignIn.graphQLType';
import UserGraphQLType from '../graphQLTypes/User.graphQLType';

const mutation = new GraphQLObjectType({
  name: 'Mutation',
  fields: {
    updateName: {
      type: UserGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
        firstName: { type: new GraphQLNonNull(GraphQLString) },
        lastName: { type: new GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parentValue, args) => {
        const { id, firstName, lastName } = args;
        return formatUser(await updateName(id, firstName, lastName));
      },
    },
    signIn: {
      type: SignInGraphQLType,
      args: {
        email: { type: new GraphQLNonNull(GraphQLString) },
        password: { type: new GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parentValue, args) => {
        const { email, password } = args;
        return {
          jwtToken: await getJWTToken(email, password),
        };
      },
    },
    uploadFile: {
      type: FileGraphQLType,
      args: {
        file: { type: new GraphQLNonNull(GraphQLUpload) },
      },
      resolve: async (parentValue, args) => {
        const { file } = args;
        const { filename } = await file;
        return { name: filename };
      },
    },
  },
});

export default mutation;
