import { combineReducers } from 'redux';
import meReducer from '../../Home/reducers/me.reducer';

const rootReducer = combineReducers({
  me: meReducer,
});

export default rootReducer;
