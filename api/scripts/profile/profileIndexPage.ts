import autocannon from 'autocannon';
import config from '../config';

const profileIndexPage = async (): Promise<autocannon.Result> => {
  return autocannon({
    connections: config.autocannon.connections,
    amount: config.autocannon.amount,
    url: config.serverURL,
    method: 'GET',
  });
};

export default profileIndexPage;
