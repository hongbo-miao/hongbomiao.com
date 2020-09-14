import { applyMiddleware, createStore } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension/developmentOnly';
import { createLogger } from 'redux-logger';
import { createEpicMiddleware } from 'redux-observable';
import rootEpic from '../epics/rootEpic';
import rootReducer from '../reducers/rootReducer';
import graphQLFetch from './graphQLFetch';

const epicMiddleware = createEpicMiddleware({
  dependencies: { fetchData: graphQLFetch },
});
const logRocketLoggerMiddleware = createLogger();

const middleware = [epicMiddleware, logRocketLoggerMiddleware];
const enhancer = applyMiddleware(...middleware);
const store = createStore(rootReducer, composeWithDevTools(enhancer));

epicMiddleware.run(rootEpic);

export default store;
