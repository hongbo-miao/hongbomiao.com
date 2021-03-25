import initUsers from './initUsers';

const initPostgres = async (): Promise<void> => {
  await initUsers();
};

export default initPostgres;
