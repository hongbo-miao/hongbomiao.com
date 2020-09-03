import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { DocumentLoad } from '@opentelemetry/plugin-document-load';
import { XMLHttpRequestPlugin } from '@opentelemetry/plugin-xml-http-request';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import { WebTracerProvider } from '@opentelemetry/web';
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
    const { token, traceURL } = Config.lightstep;
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
