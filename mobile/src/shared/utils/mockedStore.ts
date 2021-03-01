import configureStore from 'redux-mock-store';
import MeState from '../../Home/types/MeState.type';
import RootState from '../types/RootState.type';

const mockStore = configureStore();

const me: MeState = {
  name: 'Hongbo Miao',
  bio: 'Making magic happen',
};

const initialState: RootState = { me };
const mockedStore = mockStore(initialState);

export default mockedStore;
