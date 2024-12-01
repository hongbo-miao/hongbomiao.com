import meReducer, { queryMeSucceed } from './meSlice';

describe('meSlice', () => {
  const initialState = {
    name: 'Hongbo Miao',
    bio: 'Making magic happen',
  };

  test('should return the initial state', () => {
    expect(meReducer(undefined, { type: 'unknown' })).toEqual(initialState);
  });

  test('should handle queryMeSucceed', () => {
    const action = queryMeSucceed({
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
    });

    const expectedState = {
      name: 'Jack Dawson',
      bio: "I'm the king of the world!",
    };

    expect(meReducer(initialState, action)).toEqual(expectedState);
  });
});
