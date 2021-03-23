import { Epic, ofType } from 'redux-observable';
import { of } from 'rxjs';
import { AjaxError, AjaxResponse } from 'rxjs/ajax';
import { catchError, map, switchMap } from 'rxjs/operators';
import MeActionType from '../actionTypes/MeActionType';
import MeAction from '../actions/MeAction';

const queryMeEpic: Epic = (action$, state$, { fetchData }) =>
  action$.pipe(
    ofType(MeActionType.QUERY_ME),
    switchMap((action) =>
      fetchData(action.payload.query).pipe(
        map((res: AjaxResponse) => MeAction.queryMeSucceed(res)),
        catchError((err: AjaxError) => of(MeAction.queryMeFailed(err))),
      ),
    ),
  );

export default queryMeEpic;
