import opentelemetry, { Span, Tracer } from '@opentelemetry/api';

const startSpan = (tracer: Tracer): Span => {
  return tracer.startSpan('fibonacci');
};

const updateAndEndSpan = (span: Span, n: number, val: number): void => {
  span.setAttribute(String(n), val);
  span.addEvent('invoking fibonacci');
  span.end();
};

const fibonacci = (n: number, tracer: Tracer): number => {
  if (n <= 1) {
    const span = startSpan(tracer);
    const val = 1;
    updateAndEndSpan(span, n, val);
    return val;
  }

  const span = startSpan(tracer);
  const val = fibonacci(n - 1, tracer) + fibonacci(n - 2, tracer);
  updateAndEndSpan(span, n, val);
  return val;
};

const calcFibonacci = (n: number): number => {
  const tracer = opentelemetry.trace.getTracer('fibonacci-tracer');
  return fibonacci(n, tracer);
};

export default calcFibonacci;
