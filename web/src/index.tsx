// OpenTelemetry need to be setup before importing other modules
import './shared/utils/initTracer';
import './index.css';
import * as Sentry from '@sentry/react';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider } from '@tanstack/react-router';
import initSentry from './shared/utils/initSentry';
import queryClient from './shared/utils/queryClient';
import { router } from './router';

initSentry();

const container = document.getElementById('root');

const root = createRoot(container!);

root.render(
  <Sentry.ErrorBoundary fallback={<p>An error has occurred</p>}>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </Sentry.ErrorBoundary>,
);
