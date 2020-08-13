import { Observable, of } from 'rxjs';
import MeActionType from '../actionTypes/Me.actionType';
import Me from '../types/Me.type';

interface FetchMe {
  type: typeof MeActionType.FETCH_ME;
}
interface FetchMeSucceed {
  type: typeof MeActionType.FETCH_ME_SUCCEED;
  payload: {
    me: Me;
  };
}
type FetchMeFailed = Observable<{
  type: typeof MeActionType.FETCH_ME_FAILED;
  payload: Error;
}>;

const fetchMe = (): FetchMe => ({ type: MeActionType.FETCH_ME });
const fetchMeSucceed = (me: Me): FetchMeSucceed => ({ type: MeActionType.FETCH_ME_SUCCEED, payload: { me } });
const fetchMeFailed = (err: Error): FetchMeFailed => of({ type: MeActionType.FETCH_ME_FAILED, payload: err });

const MeAction = {
  fetchMe,
  fetchMeSucceed,
  fetchMeFailed,
};

export default MeAction;
