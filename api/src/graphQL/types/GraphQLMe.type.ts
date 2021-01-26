import GraphQLUser from './GraphQLUser.type';

type GraphQLMe = GraphQLUser & {
  email: string;
};

export default GraphQLMe;
