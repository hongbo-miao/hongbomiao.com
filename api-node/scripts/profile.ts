import profileIndexPage from './profile/profileIndexPage.js';
import profileMe from './profile/profileMe.js';

const promises = [profileIndexPage(), profileMe()];

Promise.all(promises).then((results) => {
  console.log(results);
});
