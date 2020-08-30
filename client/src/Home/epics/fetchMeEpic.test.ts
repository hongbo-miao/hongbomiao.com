import { StateObservable } from 'redux-observable';
import { Observable, throwError, timer } from 'rxjs';
import { AjaxResponse, AjaxError } from 'rxjs/ajax';
import { ColdObservable } from 'rxjs/internal/testing/ColdObservable';
import { switchMap } from 'rxjs/operators';
import { TestScheduler } from 'rxjs/testing';
import RootState from '../../shared/types/RootState.type';
import MeAction from '../actions/Me.action';
import meQuery from '../queries/me.query';
import fetchMeEpic from './fetchMeEpic';

describe('fetchMeEpic', () => {
  const res = {
    response: {
      data: {
        name: 'Hongbo Miao',
        bio: 'Making magic happen',
      },
    },
  } as AjaxResponse;
  const err = new Error('Test error') as AjaxError;

  test('fetchMeSucceed', () => {
    const marbles = {
      i: '-i', // Input action
      r: '--r', // Mock API response
      o: '---o', // Output action
    };

    const scheduler = new TestScheduler((actual, expected) => {
      expect(actual).toEqual(expected);
      expect(actual[0].notification.value.payload.res).toEqual(res);
    });

    scheduler.run(({ hot, cold, expectObservable }) => {
      const action$ = hot(marbles.i, {
        i: MeAction.fetchMe(meQuery),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      }) as any;
      const state$ = {} as StateObservable<RootState>;
      const dependencies = {
        fetchData: (): ColdObservable<AjaxResponse> =>
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

  test('fetchMeFailed', () => {
    const scheduler = new TestScheduler((actual, expected) => {
      expect(actual).toEqual(expected);
      expect(actual[0].notification.value.payload).toEqual(err);
    });
    const marbles = {
      i: '-i', // Input action
      d: '--|', // Mock API response time duration
      o: '---o', // Output action
    };

    scheduler.run(({ hot, expectObservable }) => {
      const action$ = hot(marbles.i, {
        i: MeAction.fetchMe(meQuery),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      }) as any;
      const state$ = {} as StateObservable<RootState>;
      const duration = scheduler.createTime(marbles.d);
      const dependencies = {
        fetchData: (): Observable<never> => timer(duration).pipe(switchMap(() => throwError(err))),
      };
      const output$ = fetchMeEpic(action$, state$, dependencies);

      expectObservable(output$).toBe(marbles.o, {
        o: MeAction.fetchMeFailed(err),
      });
    });
  });
});
