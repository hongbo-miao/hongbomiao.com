import { faker } from '@faker-js/faker';
import config from '../../../config.js';
import findUserByEmail from '../utils/findUserByEmail.js';
import formatUser from '../utils/formatUser.js';
import insertUsers from '../utils/insertUsers.js';

const initUsers = async (): Promise<void> => {
  const { email, password, firstName, lastName, bio } = config.seedUser;
  const user = formatUser(await findUserByEmail(email));
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
