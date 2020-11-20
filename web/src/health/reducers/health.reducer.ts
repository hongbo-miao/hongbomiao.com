import { Reducer } from 'redux';
import HealthActionType from '../actionTypes/Health.actionType';
import ReducerPing from '../types/ReducerHealth.type';

const initialState: ReducerPing = {};

const healthReducer: Reducer = (state = initialState, action) => {
  switch (action.type) {
    case HealthActionType.RECEIVE_PING_SUCCEED: {
      const { res } = action.payload;
      return {
        ...state,
        ping: res.data.ping,
      };
    }

    default: {
      return state;
    }
  }
};

export default healthReducer;
