#ifndef INCLUDE_APP_H_
#define INCLUDE_APP_H_

#include <signal.h>

extern volatile sig_atomic_t is_running;

void handle_signal(int signal);

#endif  // INCLUDE_APP_H_
