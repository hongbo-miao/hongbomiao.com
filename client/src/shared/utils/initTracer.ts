import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { DocumentLoad } from '@opentelemetry/plugin-document-load';
import { XMLHttpRequestPlugin } from '@opentelemetry/plugin-xml-http-request';
import { SimpleSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import { WebTracerProvider } from '@opentelemetry/web';
import { LightstepExporter } from 'lightstep-opentelemetry-exporter';
import Config from '../../Config';
import isDevelopment from './isDevelopment';
import isProduction from './isProduction';

const initTracer = (): void => {
  const serviceName = 'client-trace-service';
  const tracerProvider = new WebTracerProvider({
    // https://github.com/open-telemetry/opentelemetry-js-contrib/issues/193
    // eslint-disable-next-line @typescript-eslint/ban-ts-ignore
    // @ts-ignore
    plugins: [new DocumentLoad(), new XMLHttpRequestPlugin()],
  });

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(
        new CollectorTraceExporter({
          serviceName,
        })
      )
    );
  }

  if (isProduction()) {
    tracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(
        // eslint-disable-next-line @typescript-eslint/ban-ts-ignore
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
