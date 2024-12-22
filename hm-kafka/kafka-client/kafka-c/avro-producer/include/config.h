#ifndef INCLUDE_CONFIG_H_
#define INCLUDE_CONFIG_H_

#include <glib.h>
#include <librdkafka/rdkafka.h>

void load_config_group(rd_kafka_conf_t *conf, GKeyFile *key_file, const char *group_name);

#endif  // INCLUDE_CONFIG_H_
