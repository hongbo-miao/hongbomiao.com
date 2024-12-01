import initUsers from './initUsers.js';

const initPostgres = async (): Promise<void> => {
  await initUsers();
};

export default initPostgres;
