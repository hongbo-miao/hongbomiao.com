import { combineEpics } from 'redux-observable';
import queryMeEpic from '../../Home/epics/queryMeEpic';

const epics = [queryMeEpic];
const rootEpic = combineEpics(...epics);

export default rootEpic;
