import { combineReducers } from 'redux';
import meReducer from '../../Home/reducers/meReducer';

const rootReducer = combineReducers({
  me: meReducer,
});

export default rootReducer;
