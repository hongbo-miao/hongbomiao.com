import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { DnsInstrumentation } from '@opentelemetry/instrumentation-dns';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { GraphQLInstrumentation } from '@opentelemetry/instrumentation-graphql';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { IORedisInstrumentation } from '@opentelemetry/instrumentation-ioredis';
import { PinoInstrumentation } from '@opentelemetry/instrumentation-pino';
import { resourceFromAttributes } from '@opentelemetry/resources';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import config from '../../config.js';
import isDevelopment from '../../shared/utils/isDevelopment.js';
import isProduction from '../../shared/utils/isProduction.js';

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

  const tracerProvider = new NodeTracerProvider({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: 'hm-api-trace-service',
    }),
    spanProcessors,
  });

  tracerProvider.register();

  registerInstrumentations({
    instrumentations: [
      new DnsInstrumentation(),
      new ExpressInstrumentation(),
      new GraphQLInstrumentation(),
      new HttpInstrumentation(),
      new IORedisInstrumentation(),
      new PinoInstrumentation(),
    ],
  });
};

initTracer();
