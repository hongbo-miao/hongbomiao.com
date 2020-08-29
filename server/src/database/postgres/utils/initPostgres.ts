import insertUser from './insertUser';

const initPostgres = async (): Promise<void> => {
  await insertUser('Hongbo.Miao@example.com', '123', 'Hongbo', 'Miao');
};

export default initPostgres;
