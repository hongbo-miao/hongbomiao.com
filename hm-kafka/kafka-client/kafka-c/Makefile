CFLAGS := -Wall $(shell pkg-config --cflags glib-2.0 rdkafka)
LDLIBS := $(shell pkg-config --libs glib-2.0 rdkafka)

producer-build:
	make producer
producer-run:
	./producer config.ini

consumer-build:
	make consumer
consumer-run:
	./consumer config.ini
