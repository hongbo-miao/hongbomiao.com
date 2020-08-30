import Config from '../../../Config';
import findUserByEmail from '../utils/findUserByEmail';
import insertUser from '../utils/insertUser';

const initUsers = async (): Promise<void> => {
  const { email, password, firstName, lastName, bio } = Config.seedUser;
  if (email == null || password == null || firstName == null || lastName == null) {
    throw new Error('Missing seed user.');
  }

  const user = await findUserByEmail(email);
  if (user == null) {
    await insertUser(email, password, firstName, lastName, bio);
  }
};

export default initUsers;
