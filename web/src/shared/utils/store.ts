import { configureStore } from '@reduxjs/toolkit';
import * as Sentry from '@sentry/react';
import { createEpicMiddleware } from 'redux-observable';
import rootEpic from '../epics/rootEpic';
import rootReducer from '../reducers/rootReducer';
import graphQLFetch from './graphQLFetch';

const epicMiddleware = createEpicMiddleware({
  dependencies: { fetchData: graphQLFetch },
});

const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(epicMiddleware),
  // https://docs.sentry.io/platforms/javascript/guides/react/features/redux
  enhancers: (getDefaultEnhancers) => getDefaultEnhancers().concat(Sentry.createReduxEnhancer()),
});

epicMiddleware.run(rootEpic);

export default store;
