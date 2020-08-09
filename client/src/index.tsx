import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension/developmentOnly';
import { createEpicMiddleware } from 'redux-observable';
import HmApp from './App/components/App';
import rootEpic from './shared/epics/rootEpic';
import * as serviceWorker from './shared/libs/serviceWorker';
import rootReducer from './shared/reducers/rootReducer';
import './index.css';

const epicMiddleware = createEpicMiddleware();
const middlewares = [epicMiddleware];
const enhancer = applyMiddleware(...middlewares);
const store = createStore(rootReducer, composeWithDevTools(enhancer));

epicMiddleware.run(rootEpic);

ReactDOM.render(
  <StrictMode>
    <Provider store={store}>
      <HmApp />
    </Provider>
  </StrictMode>,
  document.getElementById('root')
);

serviceWorker.register();
