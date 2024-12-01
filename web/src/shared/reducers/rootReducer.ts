import { combineReducers } from '@reduxjs/toolkit';
import meReducer from '../../Home/slices/meSlice';
import healthReducer from '../../health/slices/healthSlice';

const rootReducer = combineReducers({
  health: healthReducer,
  me: meReducer,
});

export default rootReducer;
