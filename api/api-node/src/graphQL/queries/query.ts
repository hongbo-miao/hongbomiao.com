import { GraphQLID, GraphQLInt, GraphQLList, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/FibonacciGraphQLType.js';
import MeGraphQLType from '../graphQLTypes/MeGraphQLType.js';
import PlanetGraphQLType from '../graphQLTypes/PlanetGraphQLType.js';
import StarshipGraphQLType from '../graphQLTypes/StarshipGraphQLType.js';
import UserGraphQLType from '../graphQLTypes/UserGraphQLType.js';
import getFibonacci from '../utils/getFibonacci.js';
import getMe from '../utils/getMe.js';

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
        return dataLoaders.user.loadMany(ids);
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
        return dataLoaders.planet.loadMany(ids);
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
        return dataLoaders.starship.loadMany(ids);
      },
    },
  },
});

export default query;
