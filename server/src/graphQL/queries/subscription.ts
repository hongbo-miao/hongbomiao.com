import { GraphQLObjectType, GraphQLString } from 'graphql';
import { PubSub } from 'graphql-subscriptions';

const pubsub = new PubSub();

const subscription = new GraphQLObjectType({
  name: 'Subscription',
  fields: {
    ping: {
      type: GraphQLString,
      resolve: (source) => {
        if (source instanceof Error) {
          throw source;
        }
        return source.ping;
      },
      subscribe: () => {
        return pubsub.asyncIterator('ping');
      },
    },
  },
});

setInterval(() => {
  pubsub.publish('ping', {
    ping: 'pong',
  });
}, 5 * 1000).unref();

export default subscription;
