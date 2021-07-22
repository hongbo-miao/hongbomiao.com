import GraphQLUser from './GraphQLUser';

type GraphQLMe = GraphQLUser & {
  email: string;
};

export default GraphQLMe;
