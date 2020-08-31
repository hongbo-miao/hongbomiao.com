import faker from 'faker';
import Config from '../../../Config';
import findUserByEmail from '../utils/findUserByEmail';
import insertUsers from '../utils/insertUsers';

const initUsers = async (): Promise<void> => {
  const { email, password, firstName, lastName, bio } = Config.seedUser;
  if (email == null || password == null || firstName == null || lastName == null) {
    throw new Error('Missing seed user.');
  }

  const user = await findUserByEmail(email);
  if (user == null) {
    let users = [{ email, password, firstName, lastName, bio }];
    for (let i = 0; i < 20; i += 1) {
      users = [
        ...users,
        {
          email: faker.internet.email(),
          password: faker.internet.password(),
          firstName: faker.name.firstName(),
          lastName: faker.name.lastName(),
          bio: faker.lorem.sentence(),
        },
      ];
    }
    await insertUsers(users);
  }
};

export default initUsers;
