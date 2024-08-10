#include <avro.h>
#include <glib.h>
#include <librdkafka/rdkafka.h>
#include <libserdes/serdes-avro.h>
#include <libserdes/serdes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.c"

#define ARR_SIZE(arr) (sizeof((arr)) / sizeof((arr[0])))

static volatile bool is_running = true;

static void delivery_report(rd_kafka_t *kafka_handle,
                            const rd_kafka_message_t *rkmessage, void *opaque) {
  if (rkmessage->err) {
    g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
  }
}

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    is_running = false;
  }
}

int main(int argc, char **argv) {
  rd_kafka_t *producer;
  rd_kafka_conf_t *conf;
  serdes_conf_t *serdes_conf;
  serdes_t *serdes;
  char errstr[512];

  if (argc != 2) {
    g_error("Usage: %s <config.ini>", argv[0]);
    return 1;
  }

  const char *config_file = argv[1];

  g_autoptr(GError) error = NULL;
  g_autoptr(GKeyFile) key_file = g_key_file_new();
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE,
                                 &error)) {
    g_error("Error loading config file: %s", error->message);
    return 1;
  }

  conf = rd_kafka_conf_new();
  load_config_group(conf, key_file, "default");

  rd_kafka_conf_set(conf, "queue.buffering.max.messages", "10000000", NULL, 0);
  rd_kafka_conf_set(conf, "queue.buffering.max.kbytes", "10485760", NULL, 0);
  rd_kafka_conf_set(conf, "batch.size", "65536", NULL, 0);
  rd_kafka_conf_set(conf, "linger.ms", "5", NULL, 0);
  rd_kafka_conf_set_dr_msg_cb(conf, delivery_report);

  producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!producer) {
    g_error("Failed to create new producer: %s", errstr);
    return 1;
  }

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  serdes_conf = serdes_conf_new(
      NULL, 0, "schema.registry.url",
      "https://hm-confluent-schema-registry.internal.hongbomiao.com", NULL);

  serdes = serdes_new(serdes_conf, errstr, sizeof(errstr));
  if (!serdes) {
    g_error("Failed to create serdes instance: %s", errstr);
    return 1;
  }

  const char *topic = "production.iot.device.json";
  const char *schema_name = "production.iot.device.json-value";
  serdes_schema_t *schema =
      serdes_schema_get(serdes, schema_name, -1, errstr, sizeof(errstr));
  if (!schema) {
    g_error("Failed to retrieve AVRO schema: %s", errstr);
    return 1;
  }

  const char *device_ids[6] = {"device1", "device2", "device3",
                               "device4", "device5", "device6"};
  const char *status_list[3] = {"online", "offline", "maintenance"};
  const char *locations[3] = {"locationA", "locationB", "locationC"};
  const char *types[3] = {"type1", "type2", "type3"};

  srandom(time(NULL));  // Seed the random number generator

  while (is_running) {
    const char *key = device_ids[random() % ARR_SIZE(device_ids)];

    const char *status = status_list[random() % ARR_SIZE(status_list)];
    const char *location = locations[random() % ARR_SIZE(locations)];
    const char *type = types[random() % ARR_SIZE(types)];
    double temperature = ((double)random() / RAND_MAX) * 100.0 - 50.0;
    double humidity = ((double)random() / RAND_MAX);
    int battery = random() % 101;
    int signal_strength = random() % 101;
    const char *mode = (random() % 2) ? "manual" : "auto";
    bool active = (random() % 2);

    avro_schema_t serdes_schema = serdes_schema_avro(schema);
    avro_value_iface_t *record_class =
        avro_generic_class_from_schema(serdes_schema);

    avro_value_t record;
    avro_generic_value_new(record_class, &record);

    avro_value_t field;
    if (avro_value_get_by_name(&record, "status", &field, NULL) == 0) {
      avro_value_set_string(&field, status);
    }
    if (avro_value_get_by_name(&record, "location", &field, NULL) == 0) {
      avro_value_set_string(&field, location);
    }
    if (avro_value_get_by_name(&record, "type", &field, NULL) == 0) {
      avro_value_set_string(&field, type);
    }
    if (avro_value_get_by_name(&record, "temperature", &field, NULL) == 0) {
      avro_value_set_long(&field, temperature);
    }
    if (avro_value_get_by_name(&record, "humidity", &field, NULL) == 0) {
      avro_value_set_double(&field, humidity);
    }
    if (avro_value_get_by_name(&record, "battery", &field, NULL) == 0) {
      avro_value_set_long(&field, battery);
    }
    if (avro_value_get_by_name(&record, "signal_strength", &field, NULL) == 0) {
      avro_value_set_long(&field, signal_strength);
    }
    if (avro_value_get_by_name(&record, "mode", &field, NULL) == 0) {
      avro_value_set_string(&field, mode);
    }
    if (avro_value_get_by_name(&record, "active", &field, NULL) == 0) {
      avro_value_set_boolean(&field, active);
    }

    void *avro_payload = NULL;
    size_t avro_size;
    serdes_err_t serr = serdes_schema_serialize_avro(
        schema, &record, &avro_payload, &avro_size, errstr, sizeof(errstr));
    if (serr != SERDES_ERR_OK) {
      g_error("Failed to serialize data: %s", serdes_err2str(serr));
      return 1;
    }

    rd_kafka_resp_err_t err;
    err = rd_kafka_producev(producer, RD_KAFKA_V_TOPIC(topic),
                            RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                            RD_KAFKA_V_KEY((void *)key, strlen(key)),
                            RD_KAFKA_V_VALUE(avro_payload, avro_size),
                            RD_KAFKA_V_OPAQUE(NULL), RD_KAFKA_V_END);

    if (err) {
      g_error("Failed to produce to topic %s: %s", topic,
              rd_kafka_err2str(err));
      return 1;
    }

    free(avro_payload);
    avro_value_decref(&record);
    rd_kafka_poll(producer, 0);
    g_usleep(50);  // μs
  }

  g_message("Flushing final messages ...");
  rd_kafka_flush(producer, 10 * 1000);

  if (rd_kafka_outq_len(producer) > 0) {
    g_error("%d message(s) were not delivered", rd_kafka_outq_len(producer));
  }

  g_message("Producer stopped.");
  rd_kafka_destroy(producer);
  serdes_schema_destroy(schema);
  serdes_destroy(serdes);
  return 0;
}
