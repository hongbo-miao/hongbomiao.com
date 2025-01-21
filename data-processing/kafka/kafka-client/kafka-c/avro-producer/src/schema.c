#include "../include/schema.h"

void print_serdes_schema(serdes_schema_t *serdes_schema) {
  if (!serdes_schema) {
    g_print("serdes_schema is NULL.\n");
    return;
  }

  const char *schema_name = serdes_schema_name(serdes_schema);
  if (schema_name) {
    g_print("schema_name: %s\n", schema_name);
  } else {
    g_print("Failed to retrieve schema_name.\n");
  }

  int schema_id = serdes_schema_id(serdes_schema);
  g_print("schema_id: %d\n", schema_id);

  const char *schema_definition = serdes_schema_definition(serdes_schema);
  if (schema_definition) {
    g_print("serdes_schema: %s\n", schema_definition);
  } else {
    g_print("Failed to retrieve serdes_schema.\n");
  }
}

void print_avro_schema(avro_schema_t schema) {
  char schema_str[8192];
  avro_writer_t writer = avro_writer_memory(schema_str, sizeof(schema_str));
  avro_schema_to_json(schema, writer);
  g_print("schema: %s\n", schema_str);
}
