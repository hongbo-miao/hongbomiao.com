import GraphQLResponse from '../../shared/types/GraphQLResponse';
import {
  subscribePing as subscribePingSliceAction,
  receivePingSucceed as receivePingSucceedSliceAction,
  receivePingFailed as receivePingFailedSliceAction,
} from '../slices/healthSlice';
import GraphQLPing from '../types/GraphQLPing';

type SubscribePing = ReturnType<typeof subscribePingSliceAction>;
type ReceivePingSucceed = ReturnType<typeof receivePingSucceedSliceAction>;
type ReceivePingFailed = ReturnType<typeof receivePingFailedSliceAction>;

const subscribePing = (query: string): SubscribePing => subscribePingSliceAction({ query });
const receivePingSucceed = (res: GraphQLResponse<GraphQLPing>): ReceivePingSucceed =>
  receivePingSucceedSliceAction({ res });
const receivePingFailed = (error: Error): ReceivePingFailed => receivePingFailedSliceAction({ error });

const HealthAction = {
  subscribePing,
  receivePingSucceed,
  receivePingFailed,
};

export default HealthAction;
