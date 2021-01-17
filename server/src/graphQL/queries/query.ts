import { GraphQLID, GraphQLInt, GraphQLList, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/Fibonacci.graphQLType';
import MeGraphQLType from '../graphQLTypes/Me.graphQLType';
import PlanetGraphQLType from '../graphQLTypes/Planet.graphQLType';
import StarshipGraphQLType from '../graphQLTypes/Starship.graphQLType';
import UserGraphQLType from '../graphQLTypes/User.graphQLType';
import getFibonacci from '../utils/getFibonacci';
import getMe from '../utils/getMe';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    fibonacci: {
      type: FibonacciGraphQLType,
      args: {
        n: { type: new GraphQLNonNull(GraphQLInt) },
      },
      resolve: (parentValue, args) => {
        const { n } = args;
        return getFibonacci(n);
      },
    },
    me: {
      type: MeGraphQLType,
      resolve: async () => {
        return getMe();
      },
    },
    user: {
      type: UserGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
      },
      resolve: async (parentValue, args, context) => {
        const { id } = args;
        const { dataLoaders } = context;
        return dataLoaders.user.load(id);
      },
    },
    users: {
      type: new GraphQLList(UserGraphQLType),
      args: {
        ids: { type: new GraphQLNonNull(new GraphQLList(new GraphQLNonNull(GraphQLID))) },
      },
      resolve: async (parentValue, args, context) => {
        const { ids } = args;
        const { dataLoaders } = context;
        return Promise.all(ids.map((id: string) => dataLoaders.user.load(id)));
      },
    },
    planet: {
      type: PlanetGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
      },
      resolve: async (parentValue, args, context) => {
        const { id } = args;
        const { dataLoaders } = context;
        return dataLoaders.planet.load(id);
      },
    },
    planets: {
      type: new GraphQLList(PlanetGraphQLType),
      args: {
        ids: { type: new GraphQLNonNull(new GraphQLList(new GraphQLNonNull(GraphQLID))) },
      },
      resolve: async (parentValue, args, context) => {
        const { ids } = args;
        const { dataLoaders } = context;
        return Promise.all(ids.map((id: string) => dataLoaders.planet.load(id)));
      },
    },
    starship: {
      type: StarshipGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
      },
      resolve: async (parentValue, args, context) => {
        const { id } = args;
        const { dataLoaders } = context;
        return dataLoaders.starship.load(id);
      },
    },
    starships: {
      type: new GraphQLList(StarshipGraphQLType),
      args: {
        ids: { type: new GraphQLNonNull(new GraphQLList(new GraphQLNonNull(GraphQLID))) },
      },
      resolve: async (parentValue, args, context) => {
        const { ids } = args;
        const { dataLoaders } = context;
        return Promise.all(ids.map((id: string) => dataLoaders.starship.load(id)));
      },
    },
  },
});

export default query;
