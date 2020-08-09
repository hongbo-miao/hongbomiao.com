import configureStore from 'redux-mock-store';
import Me from '../../Home/types/Me.type';
import RootState from '../types/RootState.type';

const mockStore = configureStore();
const me: Me = {
  name: 'Hongbo Miao',
  slogan: 'Making magic happen',
};
const initialState: RootState = { me };
const mockedStore = mockStore(initialState);

export default mockedStore;
