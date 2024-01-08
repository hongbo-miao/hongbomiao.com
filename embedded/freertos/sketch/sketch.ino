#include <Arduino_FreeRTOS.h>

#define RED 6
#define BLUE 7
#define YELLOW 8

const TickType_t _500ms = pdMS_TO_TICKS(500);

void setup() {
  xTaskCreate(redLedControllerTask, "RED LED Task", 128, NULL, 1, NULL);
  xTaskCreate(blueLedControllerTask, "BLUE LED Task", 128, NULL, 1, NULL);
  xTaskCreate(yellowLedControllerTask, "YELLOW LED Task", 128, NULL, 1, NULL);
}

void redLedControllerTask(void *pvParameters) {
  pinMode(RED, OUTPUT);

  while (1) {
    digitalWrite(RED, digitalRead(RED) ^ 1);
    vTaskDelay(_500ms);
  }
}

void blueLedControllerTask(void *pvParameters) {
  pinMode(BLUE, OUTPUT);

  while (1) {
    digitalWrite(BLUE, digitalRead(BLUE) ^ 1);
    vTaskDelay(_500ms);
  }
}

void yellowLedControllerTask(void *pvParameters) {
  pinMode(YELLOW, OUTPUT);

  while (1) {
    digitalWrite(YELLOW, digitalRead(YELLOW) ^ 1);
    vTaskDelay(_500ms);
  }
}

void loop() {}
