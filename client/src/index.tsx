import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension';
import { createEpicMiddleware } from 'redux-observable';
import HmApp from './App/components/App';
import rootEpic from './Home/epics/getMeEpic';
import rootReducer from './reducer';
import * as serviceWorker from './shared/lib/serviceWorker';
import isProd from './shared/utils/isProd';
import './index.css';

const epicMiddleware = createEpicMiddleware();
const middlewares = [epicMiddleware];
const enhancer = applyMiddleware(...middlewares);
const store = createStore(rootReducer, isProd ? enhancer : composeWithDevTools(enhancer));

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
