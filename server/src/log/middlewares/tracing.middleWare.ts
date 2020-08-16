import opentelemetry from '@opentelemetry/api';
import { NextFunction, Request, RequestHandler, Response } from 'express';

const tracingMiddleWare = (): RequestHandler => {
  return (req: Request, res: Response, next: NextFunction) => {
    const { path } = req;
    const tracer = opentelemetry.trace.getTracer('hongbomiao-server');
    const span = tracer.startSpan(path);

    // // Extracting the tracing headers from the incoming http request
    // const wireCtx = tracer.extract(opentracing.FORMAT_HTTP_HEADERS, req.headers);
    // // Creating our span with context from incoming request
    // // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // // @ts-ignore
    // const span = tracer.startSpan(req.path, { childOf: wireCtx });
    // // Use the log api to capture a log
    // span.log({ event: 'request_received' });
    //
    // // Use the setTag api to capture standard span tags for http traces
    // span.setTag(opentracing.Tags.HTTP_METHOD, req.method);
    // span.setTag(opentracing.Tags.SPAN_KIND, opentracing.Tags.SPAN_KIND_RPC_SERVER);
    // span.setTag(opentracing.Tags.HTTP_URL, req.path);
    //
    // /*
    //  * include trace ID in headers so that we can debug slow requests we see in
    //  * the browser by looking up the trace ID found in response headers
    //  */
    // const responseHeaders = {};
    // tracer.inject(span, opentracing.FORMAT_HTTP_HEADERS, responseHeaders);
    // res.set(responseHeaders);
    //
    // // add the span to the request object for any other handler to use the span
    // Object.assign(req, { span });
    // res.on('close', () => {
    //   span.end();
    // });
    // finalize the span when the response is completed
    res.on('finish', () => {
      span.end();
    });
    next();
  };
};

export default tracingMiddleWare;
