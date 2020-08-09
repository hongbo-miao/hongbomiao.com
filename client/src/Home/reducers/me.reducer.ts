import { Reducer } from 'redux';
import MeActionType from '../actionTypes/me.actionType';
import Me from '../types/me.type';

const initialState: Me = {
  name: 'Hongbo Miao',
  slogan: 'Making magic happen',
};

const meReducer: Reducer = (state = initialState, action) => {
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
