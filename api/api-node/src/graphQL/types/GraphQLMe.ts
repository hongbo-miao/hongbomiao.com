import GraphQLUser from './GraphQLUser.js';

type GraphQLMe = GraphQLUser & {
  email: string;
};

export default GraphQLMe;
