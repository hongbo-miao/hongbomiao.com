import { CollectorTraceExporter } from '@opentelemetry/exporter-collector';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load';
import { XMLHttpRequestInstrumentation } from '@opentelemetry/instrumentation-xml-http-request';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/tracing';
import { WebTracerProvider } from '@opentelemetry/web';
import config from '../../config';
import isDevelopment from './isDevelopment';
import isProduction from './isProduction';

const initTracer = (): void => {
  const tracerProvider = new WebTracerProvider({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: 'hm-web-trace-service',
    }),
  });

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new CollectorTraceExporter()));
  }

  if (isProduction()) {
    const { token, traceURL } = config.lightstep;
    tracerProvider.addSpanProcessor(
      new BatchSpanProcessor(
        new CollectorTraceExporter({
          url: traceURL,
          headers: {
            'Lightstep-Access-Token': token,
          },
        }),
      ),
    );
  }

  tracerProvider.register();

  registerInstrumentations({
    instrumentations: [new DocumentLoadInstrumentation(), new XMLHttpRequestInstrumentation()],
  });
};

initTracer();
