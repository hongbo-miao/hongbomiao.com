import { combineReducers } from 'redux';
import meReducer from '../../Home/reducers/me.reducer';
import healthReducer from '../../health/reducers/health.reducer';

const rootReducer = combineReducers({
  health: healthReducer,
  me: meReducer,
});

export default rootReducer;
