type GraphQLPlanet = {
  id: string;
  name: string;
  rotationPeriod: number | null;
  orbitalPeriod: number | null;
  diameter: number | null;
  climates: string[];
  gravity: string;
  terrains: string[];
  surfaceWater: number | null;
  population: number | null;
};

export default GraphQLPlanet;
