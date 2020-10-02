import configureStore from 'redux-mock-store';
import ReducerMe from '../../Home/types/ReducerMe.type';
import ReducerHealth from '../../health/types/ReducerHealth.type';
import RootState from '../types/RootState.type';

const mockStore = configureStore();

const me: ReducerMe = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};
const health: ReducerHealth = {};

const initialState: RootState = { me, health };
const mockedStore = mockStore(initialState);

export default mockedStore;
