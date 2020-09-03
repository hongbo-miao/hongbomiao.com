import { CollectorMetricExporter } from '@opentelemetry/exporter-collector';
import { MeterProvider } from '@opentelemetry/metrics';

const metricExporter = new CollectorMetricExporter({
  serviceName: 'server-metric-service',
});

const meter = new MeterProvider({
  exporter: metricExporter,
  interval: 1000,
}).getMeter('server-meter');

export default meter;
