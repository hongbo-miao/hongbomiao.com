import { Reducer } from 'redux';
import MeActionType from '../actionTypes/me.actionType';

const meReducer: Reducer = (state = {}, action) => {
  switch (action.type) {
    case MeActionType.GET_ME_SUCCEED: {
      const { me } = action.payload;
      return {
        ...state,
        ...me,
      };
    }

    default: {
      return state;
    }
  }
};

export default meReducer;
