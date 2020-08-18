import { DocumentLoad } from '@opentelemetry/plugin-document-load';
import { XMLHttpRequestPlugin } from '@opentelemetry/plugin-xml-http-request';
import { SimpleSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import { WebTracerProvider } from '@opentelemetry/web';
import { LightstepExporter } from 'lightstep-opentelemetry-exporter';
import Config from '../../Config';
import isProduction from './isProduction';

const initTracer = (): void => {
  const serviceName = 'hongbomiao-client';
  const tracerProvider = new WebTracerProvider({
    plugins: [new DocumentLoad(), new XMLHttpRequestPlugin()],
  });

  if (!isProduction()) {
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
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
