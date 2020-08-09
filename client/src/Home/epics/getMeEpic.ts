import { Epic, ofType } from 'redux-observable';
import { ajax } from 'rxjs/ajax';
import { catchError, map, mergeMap } from 'rxjs/operators';
import Config from '../../Config';
import MeActionType from '../actionTypes/Me.actionType';
import MeAction from '../actions/Me.action';
import meQuery from '../queries/me.query';

const getMeEpic: Epic = (action$) =>
  action$.pipe(
    ofType(MeActionType.GET_ME),
    mergeMap(() =>
      ajax
        .post(
          Config.graphQLURL,
          {
            query: meQuery,
          },
          { 'Content-Type': 'application/json' }
        )
        .pipe(
          map((res) => MeAction.getMeSucceed(res.response.data.me)),
          catchError(MeAction.getMeFailed)
        )
    )
  );

export default getMeEpic;
