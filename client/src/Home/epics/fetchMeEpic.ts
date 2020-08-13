import { Epic, ofType } from 'redux-observable';
import { AjaxResponse } from 'rxjs/ajax';
import { catchError, map, mergeMap } from 'rxjs/operators';
import MeActionType from '../actionTypes/Me.actionType';
import MeAction from '../actions/Me.action';

const fetchMeEpic: Epic = (action$, state$, { fetchData }) =>
  action$.pipe(
    ofType(MeActionType.FETCH_ME),
    mergeMap((action) =>
      fetchData(action.payload.query).pipe(
        map((res: AjaxResponse) => MeAction.fetchMeSucceed(res.response.data.me)),
        catchError(MeAction.fetchMeFailed)
      )
    )
  );

export default fetchMeEpic;
