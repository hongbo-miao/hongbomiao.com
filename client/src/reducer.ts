import { combineReducers } from 'redux';
import meReducer from './Home/reducers/me.reducer';

const rootReducer = combineReducers({
  me: meReducer,
});

export type RootState = ReturnType<typeof rootReducer>;
export default rootReducer;
