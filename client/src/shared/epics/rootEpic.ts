import { combineEpics } from 'redux-observable';
import fetchMeEpic from '../../Home/epics/fetchMeEpic';

const rootEpic = combineEpics(fetchMeEpic);

export default rootEpic;
