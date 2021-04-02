import DataLoader from 'dataloader';
import planetDataLoader from './planetDataLoader';
import starshipDataLoader from './starshipDataLoader';
import userDataLoader from './userDataLoader';

const dataLoaders = (): Record<string, DataLoader<string, unknown>> => ({
  planet: planetDataLoader(),
  starship: starshipDataLoader(),
  user: userDataLoader(),
});

export default dataLoaders;
