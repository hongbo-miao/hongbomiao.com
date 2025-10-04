import { GraphQLFloat, GraphQLID, GraphQLInt, GraphQLList, GraphQLObjectType, GraphQLString } from 'graphql';

const PlanetGraphQLType = new GraphQLObjectType({
  name: 'Planet',
  fields: {
    id: { type: GraphQLID },
    name: { type: GraphQLString },
    rotationPeriod: { type: GraphQLInt },
    orbitalPeriod: { type: GraphQLInt },
    diameter: { type: GraphQLInt },
    climates: { type: new GraphQLList(GraphQLString) },
    gravity: { type: GraphQLString },
    terrains: { type: new GraphQLList(GraphQLString) },
    surfaceWater: { type: GraphQLFloat },
    population: { type: GraphQLFloat },
  },
});

export default PlanetGraphQLType;
