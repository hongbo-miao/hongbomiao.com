import GraphQLStarship from '../../../graphQL/types/GraphQLStarship.js';
import SWAPIStarship from '../types/SWAPIStarship.js';
import convertToNumber from './convertToNumber.js';

const formatStarship = (id: string, starship: SWAPIStarship): GraphQLStarship | null => {
  if (starship == null) return null;

  const {
    name,
    model,
    manufacturer,
    cost_in_credits: costInCredits,
    length,
    max_atmosphering_speed: speed,
    cargo_capacity: cargoCapacity,
    starship_class: starshipClass,
  } = starship;

  return {
    id,
    name,
    model,
    manufacturer,
    costInCredits: convertToNumber(costInCredits),
    length: convertToNumber(length),
    speed: convertToNumber(speed),
    cargoCapacity: convertToNumber(cargoCapacity),
    starshipClass,
  };
};

export default formatStarship;
