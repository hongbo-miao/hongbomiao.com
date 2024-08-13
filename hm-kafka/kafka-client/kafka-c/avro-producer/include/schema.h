#ifndef SCHEMA_H
#define SCHEMA_H

#include <avro.h>
#include <glib.h>
#include <libserdes/serdes.h>

void print_serdes_schema(serdes_schema_t *serdes_schema);
void print_avro_schema(avro_schema_t schema);

#endif
