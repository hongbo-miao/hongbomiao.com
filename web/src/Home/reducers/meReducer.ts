import { Reducer } from 'redux';
import MeActionType from '../actionTypes/MeActionType';
import MeState from '../types/MeState';

const initialState: MeState = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};

// eslint-disable-next-line default-param-last
const meReducer: Reducer = (state = initialState, action) => {
  switch (action.type) {
    case MeActionType.QUERY_ME_SUCCEED: {
      const { res } = action.payload;
      return {
        ...state,
        ...(res.response.data.me || {}),
      };
    }

    default: {
      return state;
    }
  }
};

export default meReducer;
