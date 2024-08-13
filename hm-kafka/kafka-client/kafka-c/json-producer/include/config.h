#ifndef CONFIG_H
#define CONFIG_H

#include <glib.h>
#include <librdkafka/rdkafka.h>

void load_config_group(rd_kafka_conf_t *conf, GKeyFile *key_file, const char *group_name);

#endif
