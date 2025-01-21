#ifndef INCLUDE_RECORD_H_
#define INCLUDE_RECORD_H_

#include <avro.h>
#include <glib.h>

int set_boolean_field(avro_value_t *record, const char *field_name, bool value);
int set_double_field(avro_value_t *record, const char *field_name, double value);
int set_long_field(avro_value_t *record, const char *field_name, int64_t value);
int set_string_field(avro_value_t *record, const char *field_name, const char *value);
void print_record(avro_value_t *record);

#endif  // INCLUDE_RECORD_H_
