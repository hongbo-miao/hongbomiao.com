import { Epic, ofType } from 'redux-observable';
import { catchError, map, mergeMap } from 'rxjs/operators';
import graphQLFetch from '../../shared/utils/graphQLFetch';
import MeActionType from '../actionTypes/Me.actionType';
import MeAction from '../actions/Me.action';
import meQuery from '../queries/me.query';

const fetchMeEpic: Epic = (action$) =>
  action$.pipe(
    ofType(MeActionType.FETCH_ME),
    mergeMap(() =>
      graphQLFetch(meQuery).pipe(
        map((res) => MeAction.fetchMeSucceed(res.response.data.me)),
        catchError(MeAction.fetchMeFailed)
      )
    )
  );

export default fetchMeEpic;
