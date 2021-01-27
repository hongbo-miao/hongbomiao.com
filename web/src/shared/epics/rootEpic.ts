import { combineEpics } from 'redux-observable';
import queryMeEpic from '../../Home/epics/queryMeEpic';
import pingEpic from '../../health/epics/pingEpic';

const epics = [queryMeEpic, pingEpic];
const rootEpic = combineEpics(...epics);

export default rootEpic;
