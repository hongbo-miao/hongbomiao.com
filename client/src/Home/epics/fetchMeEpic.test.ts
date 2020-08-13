import { AjaxResponse } from 'rxjs/ajax';
import { TestScheduler } from 'rxjs/testing';
import MeAction from '../actions/Me.action';
import meQuery from '../queries/me.query';
import fetchMeEpic from './fetchMeEpic';

describe('fetchMeEpic', () => {
  test('fetchMeSucceed', () => {
    const res = {
      response: {
        data: {
          name: 'Hongbo Miao',
          slogan: 'Making magic happen',
        },
      },
    } as AjaxResponse;
    const marbles = {
      i: '-i', // input action
      r: '--r', // mock api response
      o: '---o', // output action
    };

    const testScheduler = new TestScheduler((actual, expected) => {
      expect(actual).toEqual(expected);
      expect(actual[0].notification.value.payload.res).toEqual(res);
    });

    testScheduler.run(({ hot, cold, expectObservable }) => {
      const action$ = hot(marbles.i, {
        i: MeAction.fetchMe(meQuery),
      }) as any;
      const state$ = null as any;
      const dependencies = {
        fetchData: () =>
          cold(marbles.r, {
            r: res,
          }),
      };

      const output$ = fetchMeEpic(action$, state$, dependencies);
      expectObservable(output$).toBe(marbles.o, {
        o: MeAction.fetchMeSucceed(res),
      });
    });
  });
});
