import { combineEpics } from 'redux-observable';
import getMeEpic from '../../Home/epics/getMeEpic';

const rootEpic = combineEpics(getMeEpic);

export default rootEpic;
