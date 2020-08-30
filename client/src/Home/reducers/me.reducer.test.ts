import { Action } from 'redux';
import MeActionType from '../actionTypes/Me.actionType';
import meReducer from './me.reducer';

describe('meReducer', () => {
  test('initial state', () => {
    expect(meReducer(undefined, {} as Action)).toEqual({
      name: 'Hongbo Miao',
      bio: 'Making magic happen',
    });
  });

  test('handle FETCH_ME_SUCCEED', () => {
    expect(
      meReducer([], {
        type: MeActionType.FETCH_ME_SUCCEED,
        payload: {
          res: {
            response: {
              data: {
                me: {
                  name: 'Jack Dawson',
                  bio: "I'm the king of the world!",
                },
              },
            },
          },
        },
      })
    ).toEqual({
      name: 'Jack Dawson',
      bio: "I'm the king of the world!",
    });
  });
});
