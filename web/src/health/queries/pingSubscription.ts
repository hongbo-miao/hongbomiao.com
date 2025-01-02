import { print, parse } from 'graphql';

const pingSubscription = print(
  parse(`
    subscription {
      ping
    }
  `),
);

export default pingSubscription;
