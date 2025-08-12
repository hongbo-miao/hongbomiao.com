// OpenTelemetry need to be setup before importing other modules
import './shared/utils/initTracer';
import './index.css';
import './shared/styles/shadcn.css';
import * as Sentry from '@sentry/react';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { createRoot } from 'react-dom/client';
import HmApp from './App/components/App';
import initSentry from './shared/utils/initSentry';
import queryClient from './shared/utils/queryClient';

initSentry();

const container = document.getElementById('root');

const root = createRoot(container!);

root.render(
  <Sentry.ErrorBoundary fallback={<p>An error has occurred</p>}>
    <QueryClientProvider client={queryClient}>
      <HmApp />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </Sentry.ErrorBoundary>,
);
