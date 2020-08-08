import { Epic, ofType } from 'redux-observable';
import { ajax } from 'rxjs/ajax';
import { catchError, map, mergeMap } from 'rxjs/operators';
import config from '../../config';
import MeActionTypes from '../actionTypes/me.actionType';
import MeActions from '../actions/me.action';
import meQuery from '../queries/me.query';

const getMeEpic: Epic = (action$) =>
  action$.pipe(
    ofType(MeActionTypes.GET_ME),
    mergeMap(() =>
      ajax
        .post(
          config.graphQLURL,
          {
            query: meQuery,
          },
          { 'Content-Type': 'application/json' }
        )
        .pipe(
          map((res) => MeActions.getMeSucceed(res.response.data.me)),
          catchError(MeActions.getMeFailed)
        )
    )
  );

export default getMeEpic;
