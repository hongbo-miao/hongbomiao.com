#include "../include/app.h"

volatile sig_atomic_t is_running = true;

void handle_signal(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    is_running = false;
  }
}
