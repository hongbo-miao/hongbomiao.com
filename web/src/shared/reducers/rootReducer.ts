import { combineReducers } from 'redux';
import meReducer from '../../Home/reducers/meReducer';
import healthReducer from '../../health/reducers/healthReducer';

const rootReducer = combineReducers({
  health: healthReducer,
  me: meReducer,
});

export default rootReducer;
