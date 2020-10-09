import autocannon from 'autocannon';

async function profile() {
  const result = await autocannon({
    url: 'http://localhost:5000',
    connections: 5,
    amount: 500,
  });

  // eslint-disable-next-line no-console
  console.log(result);
}

profile();
