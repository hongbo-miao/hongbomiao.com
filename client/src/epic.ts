import { combineEpics } from 'redux-observable';
import getMeEpic from './Home/epics/getMeEpic';

export default combineEpics(getMeEpic);
