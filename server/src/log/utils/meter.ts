import { CollectorMetricExporter } from '@opentelemetry/exporter-collector';
import { MeterProvider } from '@opentelemetry/metrics';
import isProduction from '../../shared/utils/isProduction';

const metricExporter = new CollectorMetricExporter({
  serviceName: 'server-metric-service',
});

const meter = new MeterProvider({
  exporter: metricExporter,
  interval: isProduction() ? 60 * 1000 : 5 * 1000, // ms
}).getMeter('server-meter');

export default meter;
