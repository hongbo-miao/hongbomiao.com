import { Observable, of } from 'rxjs';
import MeActionType from '../actionTypes/me.actionType';
import Me from '../types/me.type';

interface GetMe {
  type: typeof MeActionType.GET_ME;
}
interface GetMeSucceed {
  type: typeof MeActionType.GET_ME_SUCCEED;
  payload: {
    me: Me;
  };
}
type GetMeFailed = Observable<{
  type: typeof MeActionType.GET_ME_FAILED;
  payload: Error;
}>;

const getMe = (): GetMe => ({ type: MeActionType.GET_ME });
const getMeSucceed = (me: Me): GetMeSucceed => ({ type: MeActionType.GET_ME_SUCCEED, payload: { me } });
const getMeFailed = (err: Error): GetMeFailed => of({ type: MeActionType.GET_ME_FAILED, payload: err });

const MeActions = {
  getMe,
  getMeSucceed,
  getMeFailed,
};

export default MeActions;
