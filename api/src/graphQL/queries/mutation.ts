import { GraphQLID, GraphQLNonNull, GraphQLObjectType, GraphQLString } from 'graphql';
import { GraphQLUpload } from 'graphql-upload';
import formatUser from '../../database/postgres/utils/formatUser';
import updateName from '../../database/postgres/utils/updateName';
import getJWTToken from '../../security/utils/getJWTToken';
import FileGraphQLType from '../graphQLTypes/FileGraphQLType';
import SignInGraphQLType from '../graphQLTypes/SignInGraphQLType';
import UserGraphQLType from '../graphQLTypes/UserGraphQLType';

const mutation = new GraphQLObjectType({
  name: 'Mutation',
  fields: {
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
