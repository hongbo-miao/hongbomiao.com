#include <led_status.h>
#include <stdarg.h>
#include <stdio.h>

void display_status(unsigned int interval_length, t_ast_clock_tick date,
                    unsigned int date_increment, enum led_status status) {
  unsigned int i;
  unsigned int current_date = (unsigned int)date;

  for (i = 0u; i < interval_length; ++i) {
    switch (status) {
      case SET_ON:
        log_message("%4u ON    [-]\n", current_date);
        break;
      case SET_OFF:
        log_message("%4u OFF   [-]\n", current_date);
        break;
      case ON:
        log_message("%4u       [X]\n", current_date);
        break;
      case OFF:
        log_message("%4u       [ ]\n", current_date);
        break;
    }
    current_date += date_increment;
  }
}

/*
 * Simple logger function implemented using printf.
 * The call to fflush() forces the message to be printed right away.
 */
void log_message(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  fflush(stdout);
  va_end(args);
}
