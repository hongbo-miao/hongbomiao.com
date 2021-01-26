import autocannon from 'autocannon';
import config from '../config';

const profileMe = async (): Promise<autocannon.Result> => {
  const query = `
    query Me {
      me {
        name
        bio
      }
    }
  `;

  return autocannon({
    connections: config.autocannon.connections,
    amount: config.autocannon.amount,
    url: `${config.serverURL}/graphql`,
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      query,
    }),
  });
};

export default profileMe;
