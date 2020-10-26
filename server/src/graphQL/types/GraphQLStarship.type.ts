/* eslint-disable camelcase */

type GraphQLStarship = {
  id: string;
  name: string;
  model: string;
  manufacturer: string;
  costInCredits: number | null;
  length: number | null;
  speed: number | null;
  cargoCapacity: number | null;
  starshipClass: string;
};

export default GraphQLStarship;
