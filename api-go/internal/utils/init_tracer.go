package utils

import (
	"github.com/rs/zerolog/log"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func InitTracer(enableOpenTelemetryStdoutLog string, jaegerURL string) (*sdktrace.TracerProvider, error) {
	stdoutExporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Error().Err(err).Msg("stdouttrace.New")
		return nil, err
	}
	jaegerExporter, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(jaegerURL)))
	if err != nil {
		log.Error().Err(err).Msg("jaeger.New")
		return nil, err
	}

	opts := []sdktrace.TracerProviderOption{sdktrace.WithSampler(sdktrace.AlwaysSample()), sdktrace.WithBatcher(jaegerExporter)}
	if enableOpenTelemetryStdoutLog == "true" {
		opts = append(opts, sdktrace.WithBatcher(stdoutExporter))
	}
	tp := sdktrace.NewTracerProvider(opts...)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	return tp, nil
}
