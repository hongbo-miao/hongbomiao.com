import { combineEpics } from 'redux-observable';
import fetchMeEpic from '../../Home/epics/fetchMeEpic';
import pingEpic from '../../health/epics/pingEpic';

const epics = [fetchMeEpic, pingEpic];
const rootEpic = combineEpics(...epics);

export default rootEpic;
