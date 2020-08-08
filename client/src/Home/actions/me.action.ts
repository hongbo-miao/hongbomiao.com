import { Observable, of } from 'rxjs';
import MeActionTypes from '../actionTypes/me.actionType';
import Me from '../types/me.type';

interface GetMe {
  type: typeof MeActionTypes.GET_ME;
}
interface GetMeSucceed {
  type: typeof MeActionTypes.GET_ME_SUCCEED;
  payload: {
    me: Me;
  };
}
type GetMeFailed = Observable<{
  type: typeof MeActionTypes.GET_ME_FAILED;
  payload: Error;
}>;

const getMe = (): GetMe => ({ type: MeActionTypes.GET_ME });
const getMeSucceed = (me: Me): GetMeSucceed => ({ type: MeActionTypes.GET_ME_SUCCEED, payload: { me } });
const getMeFailed = (err: Error): GetMeFailed => of({ type: MeActionTypes.GET_ME_FAILED, payload: err });

const MeActions = {
  getMe,
  getMeSucceed,
  getMeFailed,
};

export default MeActions;
