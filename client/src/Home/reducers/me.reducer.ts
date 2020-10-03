import { Reducer } from 'redux';
import MeActionType from '../actionTypes/Me.actionType';
import ReducerMe from '../types/ReducerMe.type';

const initialState: ReducerMe = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};

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
