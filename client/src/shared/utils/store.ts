import { applyMiddleware, createStore } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension/developmentOnly';
import { createEpicMiddleware } from 'redux-observable';
import rootEpic from '../epics/rootEpic';
import rootReducer from '../reducers/rootReducer';
import graphQLFetch from './graphQLFetch';

const epicMiddleware = createEpicMiddleware({
  dependencies: { fetchData: graphQLFetch },
});

const middlewares = [epicMiddleware];
const enhancer = applyMiddleware(...middlewares);
const store = createStore(rootReducer, composeWithDevTools(enhancer));

epicMiddleware.run(rootEpic);

export default store;
