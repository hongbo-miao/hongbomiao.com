#include "../include/record.h"

int set_boolean_field(avro_value_t *record, const char *field_name, bool value) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, NULL) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_branch(&field, 1, &field) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_boolean(&field, value) != 0) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int set_double_field(avro_value_t *record, const char *field_name, double value) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, NULL) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_branch(&field, 1, &field) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_double(&field, value) != 0) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int set_long_field(avro_value_t *record, const char *field_name, int64_t value) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, NULL) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_branch(&field, 1, &field) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_long(&field, value) != 0) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int set_string_field(avro_value_t *record, const char *field_name, const char *value) {
  avro_value_t field;
  if (avro_value_get_by_name(record, field_name, &field, NULL) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_branch(&field, 1, &field) != 0) {
    return EXIT_FAILURE;
  }
  if (avro_value_set_string(&field, value) != 0) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

void print_record(avro_value_t *record) {
  char *record_str;
  if (avro_value_to_json(record, 1, &record_str) != 0) {
    g_printerr("Failed to convert Avro value to JSON\n");
    return;
  }
  g_print("record: %s\n", record_str);
  free(record_str);
}
