import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { GraphQLInstrumentation } from '@opentelemetry/instrumentation-graphql';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { NodeTracerProvider } from '@opentelemetry/node';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import config from '../../config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const initTracer = (): void => {
  const serviceName = 'hm-api-trace-service';
  const tracerProvider = new NodeTracerProvider();

  registerInstrumentations({
    instrumentations: [
      {
        plugins: {
          dns: {
            enabled: true,
            path: '@opentelemetry/plugin-dns',
          },
          express: {
            enabled: true,
            path: '@opentelemetry/plugin-express',
          },
          ioredis: {
            enabled: true,
            path: '@opentelemetry/plugin-ioredis',
          },
          pg: {
            enabled: true,
            path: '@opentelemetry/plugin-pg',
          },
          'pg-pool': {
            enabled: true,
            path: '@opentelemetry/plugin-pg-pool',
          },
        },
      },
    ],
    tracerProvider,
  });

  const httpInstrumentation = new HttpInstrumentation();
  httpInstrumentation.enable();

  const graphQLInstrumentation = new GraphQLInstrumentation();
  graphQLInstrumentation.enable();

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        new CollectorTraceExporter({
          serviceName,
        }),
      ),
    );
  }

  if (isProduction()) {
    const { token, traceURL } = config.lightstep;
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        new CollectorTraceExporter({
          serviceName,
          url: traceURL,
          headers: {
            'Lightstep-Access-Token': token,
          },
        }),
      ),
    );
  }

  tracerProvider.register();
};

initTracer();
