import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { DnsInstrumentation } from '@opentelemetry/instrumentation-dns';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { GraphQLInstrumentation } from '@opentelemetry/instrumentation-graphql';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { IORedisInstrumentation } from '@opentelemetry/instrumentation-ioredis';
import { PinoInstrumentation } from '@opentelemetry/instrumentation-pino';
import { Resource } from '@opentelemetry/resources';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import config from '../../config';
import isDevelopment from '../../shared/utils/isDevelopment';
import isProduction from '../../shared/utils/isProduction';

const initTracer = (): void => {
  const tracerProvider = new NodeTracerProvider({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: 'hm-api-trace-service',
    }),
  });

  if (isDevelopment()) {
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new ConsoleSpanExporter()));
    tracerProvider.addSpanProcessor(new BatchSpanProcessor(new OTLPTraceExporter()));
  }

  if (isProduction()) {
    const { token, traceURL } = config.lightstep;
    tracerProvider.addSpanProcessor(
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
