import { CollectorMetricExporter } from '@opentelemetry/exporter-collector';
import { MeterProvider } from '@opentelemetry/metrics';
import { RequestHandler } from 'express';

const metricExporter = new CollectorMetricExporter({
  serviceName: 'server-metric-service',
});

const meter = new MeterProvider({
  exporter: metricExporter,
  interval: 1000,
}).getMeter('request-meter');

const requestCounter = meter.createCounter('requestCounter', {
  description: 'Count incoming requests',
});

const requestCountMiddleware = (): RequestHandler => {
  return (req, res, next) => {
    const labels = { route: req.path };
    requestCounter.bind(labels).add(1);
    next();
  };
};

export default requestCountMiddleware;
