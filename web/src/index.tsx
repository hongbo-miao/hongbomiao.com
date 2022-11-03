// OpenTelemetry need to be setup before importing other modules
import './shared/utils/initTracer';
import './index.css';
import * as Sentry from '@sentry/react';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import HmApp from './App/components/App';
import reportWebVitals from './shared/libs/reportWebVitals';
import * as serviceWorkerRegistration from './shared/libs/serviceWorkerRegistration';
import initSentry from './shared/utils/initSentry';
import queryClient from './shared/utils/queryClient';
import store from './shared/utils/store';

initSentry();

const container = document.getElementById('root');

// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
const root = createRoot(container!);

root.render(
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  <Sentry.ErrorBoundary fallback={<p>An error has occurred</p>}>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <HmApp />
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </Provider>
  </Sentry.ErrorBoundary>,
);

/*
 * If you want your app to work offline and load faster, you can change
 * unregister() to register() below. Note this comes with some pitfalls.
 * Learn more about service workers: https://cra.link/PWA
 */
serviceWorkerRegistration.unregister();

/*
 * If you want to start measuring performance in your app, pass a function
 * to log results (for example: reportWebVitals(console.log))
 * or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
 */
reportWebVitals();
