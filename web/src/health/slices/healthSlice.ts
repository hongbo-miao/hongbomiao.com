import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import GraphQLResponse from '../../shared/types/GraphQLResponse';
import GraphQLPing from '../types/GraphQLPing';
import HealthState from '../types/HealthState';

const initialState: HealthState = {};

const healthSlice = createSlice({
  name: 'health',
  initialState,
  reducers: {
    subscribePing: {
      reducer: () => {},
      prepare: (payload: { query: string }) => ({ payload }),
    },
    receivePingSucceed: {
      reducer: (state, action: PayloadAction<{ res: GraphQLResponse<GraphQLPing> }>) => {
        const { res } = action.payload;
        if (res.data.ping == null) {
          return state;
        }
        return {
          ...state,
          ping: res.data.ping,
        };
      },
      prepare: (payload: { res: GraphQLResponse<GraphQLPing> }) => ({ payload }),
    },
    receivePingFailed: {
      reducer: () => {},
      prepare: (payload: { error: Error }) => ({ payload }),
    },
  },
});

export const { subscribePing, receivePingSucceed, receivePingFailed } = healthSlice.actions;
export default healthSlice.reducer;
