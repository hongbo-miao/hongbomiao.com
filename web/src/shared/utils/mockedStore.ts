import configureStore from 'redux-mock-store';
import MeState from '../../Home/types/MeState.type';
import HealthState from '../../health/types/HealthState.type';
import RootState from '../types/RootState.type';

const mockStore = configureStore();

const me: MeState = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};
const health: HealthState = {};

const initialState: RootState = { me, health };
const mockedStore = mockStore(initialState);

export default mockedStore;
