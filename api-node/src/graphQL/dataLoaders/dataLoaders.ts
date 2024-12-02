import DataLoader from 'dataloader';
import planetDataLoader from './planetDataLoader.js';
import starshipDataLoader from './starshipDataLoader.js';
import userDataLoader from './userDataLoader.js';

const dataLoaders = (): Record<string, DataLoader<string, unknown>> => ({
  planet: planetDataLoader(),
  starship: starshipDataLoader(),
  user: userDataLoader(),
});

export default dataLoaders;
