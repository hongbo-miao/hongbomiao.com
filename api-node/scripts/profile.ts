import profileIndexPage from './profile/profileIndexPage';
import profileMe from './profile/profileMe';

const promises = [profileIndexPage(), profileMe()];

Promise.all(promises).then((results) => {
  // eslint-disable-next-line no-console
  console.log(results);
});
