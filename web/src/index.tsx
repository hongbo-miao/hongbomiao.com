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
import initSentry from './shared/utils/initSentry';
import queryClient from './shared/utils/queryClient';
import store from './shared/utils/store';

initSentry();

const container = document.getElementById('root');

const root = createRoot(container!);

root.render(
  <Sentry.ErrorBoundary fallback={<p>An error has occurred</p>}>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <HmApp />
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </Provider>
  </Sentry.ErrorBoundary>,
);
