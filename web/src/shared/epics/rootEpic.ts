import { combineEpics } from 'redux-observable';
import queryMeEpic from '../../Home/epics/queryMe.epic';
import pingEpic from '../../health/epics/ping.epic';

const epics = [queryMeEpic, pingEpic];
const rootEpic = combineEpics(...epics);

export default rootEpic;
