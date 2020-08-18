import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { NodeTracerProvider } from '@opentelemetry/node';
import { BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor } from '@opentelemetry/tracing';
import { LightstepExporter } from 'lightstep-opentelemetry-exporter';
import Config from '../../Config';
import isDevelopment from './isDevelopment';

const initTracer = (): void => {
  const serviceName = 'hongbomiao-server';
  const tracerProvider = new NodeTracerProvider({
    plugins: {
      express: {
        enabled: true,
        path: '@opentelemetry/plugin-express',
      },
      http: {
        enabled: true,
        path: '@opentelemetry/plugin-http',
      },
      https: {
        enabled: true,
        path: '@opentelemetry/plugin-https',
      },
    },
  });

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        new JaegerExporter({
          serviceName,
          host: 'localhost',
          port: 6832,
          maxPacketSize: 65000,
        })
      )
    );
  }

  tracerProvider.addSpanProcessor(
    new SimpleSpanProcessor(
      new LightstepExporter({
        serviceName,
        token: Config.lightstepToken,
      })
    )
  );

  tracerProvider.register();
};

export default initTracer;
