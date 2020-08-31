import opentelemetry, { Span, Tracer } from '@opentelemetry/api';

const startSpan = (tracer: Tracer, parentSpan: Span | null): Span => {
  return tracer.startSpan('fibonacci', { parent: parentSpan });
};

const updateAndEndSpan = (span: Span, n: number, val: number): void => {
  span.setAttribute(String(n), val);
  span.addEvent('invoking fibonacci');
  span.end();
};

const fibonacci = (n: number, tracer: Tracer, parentSpan: Span | null): number => {
  if (n <= 1) {
    const span = startSpan(tracer, parentSpan);
    const val = 1;
    updateAndEndSpan(span, n, val);
    return val;
  }

  const span = startSpan(tracer, parentSpan);
  const val = fibonacci(n - 1, tracer, span) + fibonacci(n - 2, tracer, span);
  updateAndEndSpan(span, n, val);
  return val;
};

const calcFibonacci = (n: number): number => {
  const tracer = opentelemetry.trace.getTracer('fibonacci-tracer');
  return fibonacci(n, tracer, null);
};

export default calcFibonacci;
