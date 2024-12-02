import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.js';
import SWAPIPlanet from '../types/SWAPIPlanet.js';
import convertToNumber from './convertToNumber.js';
import splitStringToArray from './splitStringToArray.js';

const formatPlanet = (id: string, planet: SWAPIPlanet): GraphQLPlanet | null => {
  if (planet == null) return null;

  const {
    name,
    rotation_period: rotationPeriod,
    orbital_period: orbitalPeriod,
    diameter,
    climate,
    gravity,
    terrain,
    surface_water: surfaceWater,
    population,
  } = planet;

  return {
    id,
    name,
    rotationPeriod: convertToNumber(rotationPeriod),
    orbitalPeriod: convertToNumber(orbitalPeriod),
    diameter: convertToNumber(diameter),
    climates: splitStringToArray(climate),
    gravity,
    terrains: splitStringToArray(terrain),
    surfaceWater: convertToNumber(surfaceWater),
    population: convertToNumber(population),
  };
};

export default formatPlanet;
