import findUserByEmail from '../utils/findUserByEmail';
import insertUser from '../utils/insertUser';

const initUsers = async (): Promise<string | null> => {
  const users = await findUserByEmail('Hongbo.Miao@example.com');

  if (users == null || users.length === 0) {
    return insertUser('Hongbo.Miao@example.com', '123', 'Hongbo', 'Miao');
  }
  return null;
};

export default initUsers;
