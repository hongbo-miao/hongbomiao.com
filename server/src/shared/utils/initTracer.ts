import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { NodeTracerProvider } from '@opentelemetry/node';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import { LightstepExporter } from 'lightstep-opentelemetry-exporter';
import Config from '../../Config';
import isDevelopment from './isDevelopment';
import isProduction from './isProduction';

const initTracer = (): void => {
  const serviceName = 'server-trace-service';
  const tracerProvider = new NodeTracerProvider({
    plugins: {
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
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        new LightstepExporter({
          serviceName,
          token: Config.lightstepToken,
        })
      )
    );
  }

  tracerProvider.register();
};

export default initTracer;
