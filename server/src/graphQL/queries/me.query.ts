import Config from '../../Config';
import findUserByEmail from '../../database/postgres/utils/findUserByEmail';

interface Me {
  name: string;
  bio: string | null;
}

const me = async (): Promise<Me> => {
  const { email } = Config.seedUser;
  if (email == null) {
    throw new Error('Missing seed user.');
  }

  const user = await findUserByEmail(email);
  if (user == null) {
    throw new Error('User does not exist.');
  }

  const { first_name: firstName, last_name: lastName, bio } = user;
  return { name: `${firstName} ${lastName}`, bio };
};

export default me;
