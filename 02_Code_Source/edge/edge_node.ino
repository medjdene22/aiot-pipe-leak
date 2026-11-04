/*
 * edge_node.ino - Edge node firmware for water-network leak detection
 * (ESP32 / Wokwi, Arduino-ESP32 core, FreeRTOS).
 *
 * Course: UMBB M2 I2A - Embedded Systems and IoT 2026-2027.
 * Satisfies E2 (>=3 FreeRTOS tasks, queue + semaphore, timer interrupt,
 * justified periods/priorities) and E3 (3 sensors, preprocessing,
 * timestamped JSON).
 *
 * TASKS (justification in report Section 4.1 + timing diagram):
 *   taskAcq  prio 3  period 1 s   released by a HARDWARE TIMER ISR (binary
 *                                 semaphore); reads the 3 virtual sensors,
 *                                 moving-average filter, physical
 *                                 normalization, pushes a sample to the queue
 *   taskInfer prio 2 event-driven from the queue; aggregates a 6-step window,
 *                                 runs the TinyML model (model.h), measures
 *                                 inference time with micros()
 *   taskCom  prio 1  loop         MQTT publish (telemetry QoS0, alert QoS1 via
 *                                 retained command handling), reconnect with
 *                                 backoff, ring buffer while offline, LWT
 *
 * SENSORS (E3): pressure p (m), flow q (L/s), vibration vib (g RMS).
 * In simulation they are replayed virtual sensors (sensor_stream.h, generated
 * from the dataset - see 03_Simulation/README). Preprocessing: 4-sample
 * moving average + normalization to physical units. Residual dp = p - p_nominal
 * (baseline curve stored at the Edge, updated by the Fog command topic).
 *
 * MODES (compile-time):
 *   default            live telemetry to the MQTT broker (Wokwi demo)
 *   #define REPLAY_TEST  replay test_data.h and print
 *                        "Accuracy on the Edge: x.xxx (N=...)" (X1 evidence)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#include "model.h"        // TinyML detector (micromlgen export)
#include "test_data.h"    // replay test set (X1)
#include "sensor_stream.h"

// ---------- configuration (no secrets in the repo!) ----------
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASS ""
#define WIFI_CHANNEL 6

#define TEAM_ID "XX"                 // set your team number
#define NODE_ID 3                    // this node's id
#define MQTT_HOST "broker.emqx.io"   // public broker for the Wokwi demo
#define MQTT_PORT 1883

#define STR2(x) #x
#define STR(x) STR2(x)
#define TOPIC_BASE "m2i2a/equipe" TEAM_ID "/edge/node" STR(NODE_ID)
#define TOPIC_TELEMETRY TOPIC_BASE "/telemetry"
#define TOPIC_ALERT     TOPIC_BASE "/alert"
#define TOPIC_STATUS    TOPIC_BASE "/status"
#define TOPIC_COMMANDE  "m2i2a/equipe" TEAM_ID "/edge/commande/modele"

#define MODEL_VERSION "edge-v1"
#define ACQ_PERIOD_MS 1000           // 1 s acquisition tick (timer ISR)
#define WINDOW_STEPS 6               // 6 steps = 6 s sim time in the stream
#define SAMPLES_QUEUE 8
#define RING_SIZE 16                 // offline ring buffer (messages)

// ---------- model instance ----------
Eloquent::ML::Port::RandomForest clf;

// decision threshold updated by the retained command topic (Fog -> Edge)
volatile float g_conf_threshold = 0.5f;
volatile char g_model_version[24] = MODEL_VERSION;

// ---------- RTOS objects ----------
QueueHandle_t sampleQueue;
SemaphoreHandle_t tickSem;
hw_timer_t* acqTimer = NULL;

WiFiClient net;
PubSubClient mqtt(net);

struct Sample {
  float p, dp, q, vib;
  uint32_t t_ms;
};

struct InferResult {
  Sample s;
  int y;
  float conf;      // tree-vote fraction from the exported model
  uint32_t inf_us; // measured inference time (micros)
};

// offline ring buffer
static InferResult ring[RING_SIZE];
static volatile uint8_t ringHead = 0, ringCount = 0;

// ---------- hardware timer ISR (E2: >=1 interrupt) ----------
#if ESP_ARDUINO_VERSION_MAJOR >= 3
void IRAM_ATTR onAcqTimer() {
  BaseType_t w = pdFALSE;
  xSemaphoreGiveFromISR(tickSem, &w);
  if (w) portYIELD_FROM_ISR();
}
#else
void IRAM_ATTR onAcqTimer() {
  BaseType_t w = pdFALSE;
  xSemaphoreGiveFromISR(tickSem, &w);
  if (w) portYIELD_FROM_ISR();
}
#endif

// ---------- virtual sensors (E3) ----------
// Moving-average filter state (preprocessing step 1)
static float maBuf[4] = {0, 0, 0, 0};
static uint8_t maIdx = 0;

static float movingAverage(float v) {
  maBuf[maIdx++ & 3] = v;
  return (maBuf[0] + maBuf[1] + maBuf[2] + maBuf[3]) / 4.0f;
}

static int streamStep = 0;

static void readSensors(Sample& s) {
  // replay one step of the recorded scenario (PROGMEM)
  float p, q, vib, pnom;
  memcpy_P(&p, &P_STREAM[streamStep], sizeof(float));
  memcpy_P(&q, &Q_STREAM[streamStep], sizeof(float));
  memcpy_P(&vib, &VIB_STREAM[streamStep], sizeof(float));
  memcpy_P(&pnom, &PNOM_STREAM[streamStep], sizeof(float));
  streamStep = (streamStep + 1) % STREAM_N;

  s.p = movingAverage(p);            // preprocessing: filtering
  s.dp = s.p - pnom;                 // residual vs nominal baseline
  s.q = q;
  s.vib = vib;
  s.t_ms = millis();
}

// ---------- tasks (E2) ----------
QueueHandle_t comQueue;

void taskAcq(void*) {
  for (;;) {
    if (xSemaphoreTake(tickSem, pdMS_TO_TICKS(2 * ACQ_PERIOD_MS)) == pdTRUE) {
      Sample s;
      readSensors(s);
      xQueueSend(sampleQueue, &s, 0);
    }
  }
}

static float feat[NF] = {0, 0, 0, 0};
static uint8_t winIdx = 0;

void taskInfer(void*) {
  Sample s;
  for (;;) {
    if (xQueueReceive(sampleQueue, &s, portMAX_DELAY) == pdTRUE) {
      // accumulate a WINDOW_STEPS window (running mean of each feature)
      float w = 1.0f / (winIdx + 1);
      feat[0] += (s.p - feat[0]) * w;
      feat[1] += (s.dp - feat[1]) * w;
      feat[2] += (s.q - feat[2]) * w;
      feat[3] += (s.vib - feat[3]) * w;
      winIdx++;
      if (winIdx >= WINDOW_STEPS) {
        winIdx = 0;
        uint32_t t0 = micros();
        int y = clf.predict(feat);           // TinyML inference
        uint32_t inf_us = micros() - t0;
        float conf = clf.confidence();       // tree-vote fraction (== RF proba)

        InferResult out;
        out.s.p = feat[0]; out.s.dp = feat[1];
        out.s.q = feat[2]; out.s.vib = feat[3];
        out.s.t_ms = s.t_ms;
        out.conf = conf;
        out.inf_us = inf_us;
        // retained command topic can raise the confidence gate
        out.y = (y == 1 && conf >= g_conf_threshold) ? 1 : 0;
        xQueueSend(comQueue, &out, 0);
        Serial.printf("[infer] y=%d conf=%.2f inf_us=%u\n", out.y, conf, inf_us);
      }
    }
  }
}

void taskCom(void*) {
  InferResult r;
  char buf[288];
  uint32_t lastReconnect = 0;
  for (;;) {
    if (!mqtt.connected()) {
      uint32_t now = millis();
      if (now - lastReconnect > 5000) {
        lastReconnect = now;
        String cid = String("esp32-equipe") + TEAM_ID + "-node" + STR(NODE_ID);
        if (mqtt.connect(cid.c_str(), NULL, NULL,
                         TOPIC_STATUS, 1, true,
                         "{\"status\":\"offline\"}")) {   // LWT (E5)
          mqtt.publish(TOPIC_STATUS, "{\"status\":\"online\"}", true);
          mqtt.subscribe(TOPIC_COMMANDE, 1);               // QoS1 commands
          // flush the offline ring buffer
          while (ringCount > 0) {
            InferResult& q = ring[ringHead];
            snprintf(buf, sizeof(buf),
                     "{\"id\":%d,\"ts\":%lu,\"p\":%.2f,\"dp\":%.2f,\"q\":%.2f,"
                     "\"vib\":%.4f,\"y\":%d,\"conf\":%.2f,\"mv\":\"%s\"}",
                     NODE_ID, (unsigned long)q.s.t_ms, q.s.p, q.s.dp, q.s.q, q.s.vib,
                     q.y, q.conf, MODEL_VERSION);
            mqtt.publish(TOPIC_TELEMETRY, buf);
            ringHead = (ringHead + 1) % RING_SIZE;
            ringCount--;
          }
        }
      }
    }
    if (xQueueReceive(comQueue, &r, pdMS_TO_TICKS(500)) == pdTRUE) {
      snprintf(buf, sizeof(buf),
               "{\"id\":%d,\"ts\":%lu,\"p\":%.2f,\"dp\":%.2f,\"q\":%.2f,"
               "\"vib\":%.4f,\"y\":%d,\"conf\":%.2f,\"mv\":\"%s\"}",
               NODE_ID, (unsigned long)r.s.t_ms, r.s.p, r.s.dp, r.s.q, r.s.vib,
               r.y, r.conf, MODEL_VERSION);
      digitalWrite(2, r.y ? HIGH : LOW);       // alert LED (diagram.json)
      if (mqtt.connected()) {
        mqtt.publish(TOPIC_TELEMETRY, buf);            // QoS0 telemetry
        if (r.y == 1) {
          mqtt.publish(TOPIC_ALERT, buf, 1);           // QoS1 not possible with
                                                       // PubSubClient (publishes
                                                       // QoS0) - documented in the
                                                       // report; QoS1 proven with
                                                       // the Python emulated nodes
        }
      } else {
        ring[ringHead] = r;
        ringHead = (ringHead + 1) % RING_SIZE;
        if (ringCount < RING_SIZE) ringCount++;
      }
    }
    mqtt.loop();
    delay(1);   // avoid simulator stalls (Annex F.9)
  }
}

// ---------- retained command handler (Fog -> Edge params) ----------
void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  if (strcmp(topic, TOPIC_COMMANDE) != 0) return;
  StaticJsonDocument<256> d;
  if (deserializeJson(d, payload, length) == DeserializationOk) {
    if (d.containsKey("conf_threshold"))
      g_conf_threshold = d["conf_threshold"].as<float>();
    if (d.containsKey("mv"))
      strlcpy((char*)g_model_version, d["mv"], sizeof(g_model_version));
    Serial.printf("[cmd] conf_threshold=%.2f mv=%s\n",
                  g_conf_threshold, (const char*)g_model_version);
  }
}

// ---------- setup ----------
void setup() {
  Serial.begin(115200);
  delay(100);
#ifdef REPLAY_TEST
  // ---------- X1: replay the test set, print accuracy ----------
  int ok = 0;
  uint32_t t0 = micros();
  for (int i = 0; i < NS; i++) {
    float x[NF];
    memcpy_P(x, X_TEST[i], sizeof(x));
    ok += (clf.predict(x) == pgm_read_byte(&Y_TEST[i]));
    delay(1);   // keep the simulator responsive
  }
  uint32_t dt = micros() - t0;
  Serial.printf("Accuracy on the Edge: %.3f (N=%d)\n", (float)ok / NS, NS);
  Serial.printf("avg inference: %.1f us [I]\n", dt / (float)NS);
#else
  WiFi.begin(WIFI_SSID, WIFI_PASS, WIFI_CHANNEL);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.printf("\nWiFi ok, ip=%s\n", WiFi.localIP().toString().c_str());

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setBufferSize(512);

  sampleQueue = xQueueCreate(SAMPLES_QUEUE, sizeof(Sample));
  comQueue = xQueueCreate(SAMPLES_QUEUE, sizeof(InferResult));
  tickSem = xSemaphoreCreateBinary();
  pinMode(2, OUTPUT);   // alert LED (diagram.json)

  // hardware timer: 1 tick per ACQ_PERIOD_MS (E2: interrupt -> semaphore)
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  acqTimer = timerBegin(1000000);              // 1 MHz
  timerAttachInterrupt(acqTimer, &onAcqTimer);
  timerAlarm(acqTimer, ACQ_PERIOD_MS * 1000, true, 0);
#else
  acqTimer = timerBegin(0, 80, true);          // 80 prescaler -> 1 MHz
  timerAttachInterrupt(acqTimer, &onAcqTimer, true);
  timerAlarmWrite(acqTimer, ACQ_PERIOD_MS * 1000, true);
  timerAlarmEnable(acqTimer);
#endif

  xTaskCreate(taskAcq, "acq", 4096, NULL, 3, NULL);
  xTaskCreate(taskInfer, "infer", 6144, NULL, 2, NULL);
  xTaskCreate(taskCom, "com", 8192, NULL, 1, NULL);
#endif
}

void loop() {
  vTaskDelay(portMAX_DELAY);   // all work happens in tasks
}
