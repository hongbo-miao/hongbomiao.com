import GraphQLResponse from '../../shared/types/GraphQLResponse.type';
import HealthActionType from '../actionTypes/Health.actionType';
import GraphQLPing from '../types/GraphQLPing.type';

type SubscribePing = {
  type: typeof HealthActionType.SUBSCRIBE_PING;
  payload: {
    query: string;
  };
};
export type ReceivePingSucceed = {
  type: typeof HealthActionType.RECEIVE_PING_SUCCEED;
  payload: {
    res: GraphQLResponse<GraphQLPing>;
  };
};
type ReceivePingFailed = {
  type: typeof HealthActionType.RECEIVE_PING_FAILED;
  payload: Error;
};

const subscribePing = (query: string): SubscribePing => ({ type: HealthActionType.SUBSCRIBE_PING, payload: { query } });
const receivePingSucceed = (res: GraphQLResponse<GraphQLPing>): ReceivePingSucceed => ({
  type: HealthActionType.RECEIVE_PING_SUCCEED,
  payload: { res },
});
const receivePingFailed = (err: Error): ReceivePingFailed => ({
  type: HealthActionType.RECEIVE_PING_FAILED,
  payload: err,
});

const HealthAction = {
  subscribePing,
  receivePingSucceed,
  receivePingFailed,
};

export default HealthAction;
