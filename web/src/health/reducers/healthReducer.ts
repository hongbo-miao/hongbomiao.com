import { Reducer } from 'redux';
import HealthActionType from '../actionTypes/HealthActionType';
import HealthState from '../types/HealthState';

const initialState: HealthState = {};

// eslint-disable-next-line default-param-last
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
