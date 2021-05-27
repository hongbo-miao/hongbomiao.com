import { AjaxError, AjaxResponse } from 'rxjs/ajax';
import MeActionType from '../actionTypes/MeActionType';

type QueryMe = {
  type: typeof MeActionType.QUERY_ME;
  payload: {
    query: string;
  };
};
type QueryMeSucceed = {
  type: typeof MeActionType.QUERY_ME_SUCCEED;
  payload: {
    res: AjaxResponse<unknown>;
  };
};
type QueryMeFailed = {
  type: typeof MeActionType.QUERY_ME_FAILED;
  payload: AjaxError;
};

const queryMe = (query: string): QueryMe => ({ type: MeActionType.QUERY_ME, payload: { query } });
const queryMeSucceed = (res: AjaxResponse<unknown>): QueryMeSucceed => ({
  type: MeActionType.QUERY_ME_SUCCEED,
  payload: { res },
});
const queryMeFailed = (err: AjaxError): QueryMeFailed => ({
  type: MeActionType.QUERY_ME_FAILED,
  payload: err,
});

const MeAction = {
  queryMe,
  queryMeSucceed,
  queryMeFailed,
};

export default MeAction;
