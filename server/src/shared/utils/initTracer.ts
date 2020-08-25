import { CollectorProtocolNode, CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { NodeTracerProvider } from '@opentelemetry/node';
import { ConsoleSpanExporter, SimpleSpanProcessor } from '@opentelemetry/tracing';
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
          /\/v1\/trace/, // OpenTelemetry
        ],
        path: '@opentelemetry/plugin-http',
      },
      https: {
        enabled: true,
        // https://github.com/open-telemetry/opentelemetry-js/issues/585
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        ignoreOutgoingUrls: [
          /\/v1\/trace/, // OpenTelemetry
        ],
        path: '@opentelemetry/plugin-https',
      },
    },
  });

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(
        new CollectorTraceExporter({
          protocolNode: CollectorProtocolNode.HTTP_PROTO,
          serviceName,
        })
      )
    );
  }

  if (isProduction()) {
    tracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(
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
