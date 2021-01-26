import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { GraphQLInstrumentation } from '@opentelemetry/instrumentation-graphql';
import { NodeTracerProvider } from '@opentelemetry/node';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import config from '../../config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const initTracer = (): void => {
  const serviceName = 'hm-server-trace-service';
  const tracerProvider = new NodeTracerProvider({
    plugins: {
      dns: {
        enabled: true,
        path: '@opentelemetry/plugin-dns',
      },
      express: {
        enabled: true,
        path: '@opentelemetry/plugin-express',
      },
      http: {
        enabled: true,
        // https://github.com/open-telemetry/opentelemetry-js/issues/585
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        ignoreOutgoingUrls: [
          /\/v1\/trace/, // OpenTelemetry tracing
          /\/v1\/metrics/, // OpenTelemetry metrics
        ],
        path: '@opentelemetry/plugin-http',
      },
      https: {
        enabled: true,
        // https://github.com/open-telemetry/opentelemetry-js/issues/585
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        ignoreOutgoingUrls: [
          /\/v1\/trace/, // OpenTelemetry tracing
          /\/v1\/metrics/, // OpenTelemetry metrics
        ],
        path: '@opentelemetry/plugin-https',
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
  });

  const graphQLInstrumentation = new GraphQLInstrumentation();
  // https://github.com/open-telemetry/opentelemetry-js/issues/1724
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  graphQLInstrumentation.setTracerProvider(tracerProvider);
  graphQLInstrumentation.enable();

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        new CollectorTraceExporter({
          serviceName,
        })
      )
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
        })
      )
    );
  }

  tracerProvider.register();
};

initTracer();
