import { GraphQLFloat, GraphQLID, GraphQLInt, GraphQLObjectType, GraphQLString } from 'graphql';

const StarshipGraphQLType = new GraphQLObjectType({
  name: 'Starship',
  fields: {
    id: { type: GraphQLID },
    name: { type: GraphQLString },
    model: { type: GraphQLString },
    manufacturer: { type: GraphQLString },
    costInCredits: { type: GraphQLFloat },
    length: { type: GraphQLFloat },
    speed: { type: GraphQLInt },
    cargoCapacity: { type: GraphQLFloat },
    starshipClass: { type: GraphQLString },
  },
});

export default StarshipGraphQLType;
