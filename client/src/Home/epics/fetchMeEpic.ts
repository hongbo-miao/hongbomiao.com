import { Epic, ofType } from 'redux-observable';
import { ajax } from 'rxjs/ajax';
import { catchError, map, mergeMap } from 'rxjs/operators';
import Config from '../../Config';
import MeActionType from '../actionTypes/Me.actionType';
import MeAction from '../actions/Me.action';
import meQuery from '../queries/me.query';

const fetchMeEpic: Epic = (action$) =>
  action$.pipe(
    ofType(MeActionType.FETCH_ME),
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
          map((res) => MeAction.fetchMeSucceed(res.response.data.me)),
          catchError(MeAction.fetchMeFailed)
        )
    )
  );

export default fetchMeEpic;
