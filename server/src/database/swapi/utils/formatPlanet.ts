import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.type';
import SWAPIPlanetType from '../types/SWAPIPlanet.type';
import convertToNumber from './convertToNumber';
import splitStringToArray from './splitStringToArray';

const formatPlanet = (id: string, planet: SWAPIPlanetType): GraphQLPlanet | null => {
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
