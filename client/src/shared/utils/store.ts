import { applyMiddleware, createStore } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension/developmentOnly';
import { createEpicMiddleware } from 'redux-observable';
import rootEpic from '../epics/rootEpic';
import rootReducer from '../reducers/rootReducer';

const epicMiddleware = createEpicMiddleware();
const enhancer = applyMiddleware(epicMiddleware);
const store = createStore(rootReducer, composeWithDevTools(enhancer));

epicMiddleware.run(rootEpic);

export default store;
