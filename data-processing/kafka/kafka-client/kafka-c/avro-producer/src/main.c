#include <avro.h>
#include <glib.h>
#include <librdkafka/rdkafka.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <libserdes/serdes-avro.h>
#pragma GCC diagnostic pop
#include <libserdes/serdes.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/app.h"
#include "../include/config.h"
#include "../include/kafka.h"
#include "../include/record.h"
#include "../include/schema.h"

int main(int argc, char **argv) {
  const char *confluent_schema_registry_url =
    "https://confluent-schema-registry.internal.hongbomiao.com";
  const char *topic = "production.iot.device.avro";
  const char *schema_name = "production.iot.device.avro-value";

  rd_kafka_t *producer;
  rd_kafka_conf_t *conf;
  serdes_conf_t *serdes_conf;
  serdes_t *serdes;
  char err_str[4096];

  if (argc != 2) {
    g_error("Usage: %s <config.ini>", argv[0]);
  }

  const char *config_file = argv[1];
  g_autoptr(GError) err = NULL;
  g_autoptr(GKeyFile) key_file = g_key_file_new();
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE, &err)) {
    g_error("Failed to load config file: %s", err->message);
  }

  conf = rd_kafka_conf_new();
  load_config_group(conf, key_file, "default");
  rd_kafka_conf_set_dr_msg_cb(conf, delivery_report);
  producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, err_str, sizeof(err_str));
  conf = NULL;
  if (!producer) {
    g_error("Failed to create new producer: %s", err_str);
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  serdes_conf =
    serdes_conf_new(NULL, 0, "schema.registry.url", confluent_schema_registry_url, NULL);

  serdes = serdes_new(serdes_conf, err_str, sizeof(err_str));
  if (!serdes) {
    g_error("Failed to create serdes instance: %s", err_str);
  }

  serdes_schema_t *serdes_schema =
    serdes_schema_get(serdes, schema_name, -1, err_str, sizeof(err_str));
  if (!serdes_schema) {
    g_error("Failed to retrieve serdes_schema: %s", err_str);
  }

  print_serdes_schema(serdes_schema);
  avro_schema_t avro_schema = serdes_schema_avro(serdes_schema);
  // print_avro_schema(avro_schema);

  const char *device_ids[5] = {"device1", "device2", "device3", "device4", "device5"};
  const char *status_list[3] = {"online", "offline", "maintenance"};
  const char *locations[3] = {"locationA", "locationB", "locationC"};
  const char *types[3] = {"type1", "type2", "type3"};

  srandom(time(NULL));

  while (is_running) {
    const char *record_key = device_ids[random() % G_N_ELEMENTS(device_ids)];

    const char *status = status_list[random() % G_N_ELEMENTS(status_list)];
    const char *location = locations[random() % G_N_ELEMENTS(locations)];
    const char *type = types[random() % G_N_ELEMENTS(types)];
    double temperature = ((double)random() / RAND_MAX) * 100.0 - 50.0;
    double humidity = ((double)random() / RAND_MAX);
    int battery = random() % 101;
    int signal_strength = random() % 101;
    const char *mode = (random() % 2) ? "manual" : "auto";
    bool active = (random() % 2);

    avro_value_iface_t *record_class = avro_generic_class_from_schema(avro_schema);

    avro_value_t record;
    avro_generic_value_new(record_class, &record);

    if (set_string_field(&record, "status", status) != 0) {
      g_error("Failed to set status\n");
    }
    if (set_string_field(&record, "location", location) != 0) {
      g_error("Failed to set location\n");
    }
    if (set_string_field(&record, "type", type) != 0) {
      g_error("Failed to set type\n");
    }
    if (set_long_field(&record, "temperature", temperature) != 0) {
      g_error("Failed to set temperature\n");
    }
    if (set_double_field(&record, "humidity", humidity) != 0) {
      g_error("Failed to set humidity\n");
    }
    if (set_long_field(&record, "battery", battery) != 0) {
      g_error("Failed to set battery\n");
    }
    if (set_long_field(&record, "signal_strength", signal_strength) != 0) {
      g_error("Failed to set signal_strength\n");
    }
    if (set_string_field(&record, "mode", mode) != 0) {
      g_error("Failed to set mode\n");
    }
    if (set_boolean_field(&record, "active", active) != 0) {
      g_error("Failed to set active\n");
    }
    // print_record(&record);

    void *avro_payload = NULL;
    size_t avro_size;
    serdes_err_t serdes_err = serdes_schema_serialize_avro(serdes_schema, &record, &avro_payload,
                                                           &avro_size, err_str, sizeof(err_str));
    if (serdes_err != SERDES_ERR_OK) {
      g_error("Failed to serialize data: %s", serdes_err2str(serdes_err));
    }

    rd_kafka_resp_err_t rd_kafka_resp_err;
    rd_kafka_resp_err = rd_kafka_producev(
      producer, RD_KAFKA_V_TOPIC(topic), RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
      RD_KAFKA_V_KEY((void *)record_key, strlen(record_key)),
      RD_KAFKA_V_VALUE(avro_payload, avro_size), RD_KAFKA_V_OPAQUE(NULL), RD_KAFKA_V_END);

    if (rd_kafka_resp_err) {
      g_error("Failed to produce to topic %s: %s", topic, rd_kafka_err2str(rd_kafka_resp_err));
    }

    free(avro_payload);
    avro_value_decref(&record);
    rd_kafka_poll(producer, 0);
    g_usleep(5000);  // μs
  }

  g_message("Flushing final messages ...");
  rd_kafka_flush(producer, 10 * 1000);

  if (rd_kafka_outq_len(producer) > 0) {
    g_error("%d message(s) were not delivered", rd_kafka_outq_len(producer));
  }

  g_message("Producer stopped.");
  rd_kafka_destroy(producer);
  serdes_schema_destroy(serdes_schema);
  serdes_destroy(serdes);
  exit(EXIT_SUCCESS);
}
