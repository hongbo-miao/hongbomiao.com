import { Epic, ofType } from 'redux-observable';
import { of } from 'rxjs';
import { AjaxError, AjaxResponse } from 'rxjs/ajax';
import { catchError, map, switchMap } from 'rxjs/operators';
import MeActionType from '../actionTypes/Me.actionType';
import MeAction from '../actions/Me.action';

const fetchMeEpic: Epic = (action$, state$, { fetchData }) =>
  action$.pipe(
    ofType(MeActionType.FETCH_ME),
    switchMap((action) =>
      fetchData(action.payload.query).pipe(
        map((res: AjaxResponse) => MeAction.fetchMeSucceed(res)),
        catchError((err: AjaxError) => of(MeAction.fetchMeFailed(err)))
      )
    )
  );

export default fetchMeEpic;
