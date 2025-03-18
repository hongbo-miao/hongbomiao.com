import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load';
import { XMLHttpRequestInstrumentation } from '@opentelemetry/instrumentation-xml-http-request';
import { resourceFromAttributes } from '@opentelemetry/resources';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/sdk-trace-base';
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import config from '../../config';
import isDevelopment from './isDevelopment';
import isProduction from './isProduction';

const initTracer = (): void => {
  const spanProcessors = [];

  if (isDevelopment()) {
    spanProcessors.push(
      new BatchSpanProcessor(new ConsoleSpanExporter()),
      new BatchSpanProcessor(new OTLPTraceExporter()),
    );
  }

  if (isProduction()) {
    const { token, traceURL } = config.lightstep;
    spanProcessors.push(
      new BatchSpanProcessor(
        new OTLPTraceExporter({
          url: traceURL,
          headers: {
            'Lightstep-Access-Token': token,
          },
        }),
      ),
    );
  }

  const tracerProvider = new WebTracerProvider({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: 'hm-web-trace-service',
    }),
    spanProcessors,
  });

  tracerProvider.register();
  registerInstrumentations({
    instrumentations: [new DocumentLoadInstrumentation(), new XMLHttpRequestInstrumentation()],
  });
};

initTracer();
