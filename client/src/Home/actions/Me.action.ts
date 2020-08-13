import { AjaxError, AjaxResponse } from 'rxjs/ajax';
import MeActionType from '../actionTypes/Me.actionType';

interface FetchMe {
  type: typeof MeActionType.FETCH_ME;
  payload: {
    query: string;
  };
}
interface FetchMeSucceed {
  type: typeof MeActionType.FETCH_ME_SUCCEED;
  payload: {
    res: AjaxResponse;
  };
}
interface FetchMeFailed {
  type: typeof MeActionType.FETCH_ME_FAILED;
  payload: AjaxError;
}

const fetchMe = (query: string): FetchMe => ({ type: MeActionType.FETCH_ME, payload: { query } });
const fetchMeSucceed = (res: AjaxResponse): FetchMeSucceed => ({
  type: MeActionType.FETCH_ME_SUCCEED,
  payload: { res },
});
const fetchMeFailed = (err: AjaxError): FetchMeFailed => ({
  type: MeActionType.FETCH_ME_FAILED,
  payload: err,
});

const MeAction = {
  fetchMe,
  fetchMeSucceed,
  fetchMeFailed,
};

export default MeAction;
