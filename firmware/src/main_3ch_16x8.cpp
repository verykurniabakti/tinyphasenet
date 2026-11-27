#ifdef MODEL_3CH_16X8

#include <math.h>
#include <algorithm>
#include <vector>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "soc/rtc_cntl_reg.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TRANSPOSE_CONV kernel with the fused-activation fix
namespace tflite { TFLMRegistration Register_TRANSPOSE_CONV_ACT_FIX(); }

#ifdef MODEL_3CH_INT8
#include "phasenet_3ch_int8_model.h"
#include "hw_eval_3ch_100_int8.h"
typedef int8_t qout_t;
static const qout_t QOUT_MIN = -128;
#define QOUT_NAME "int8"
#define VARIAN_NAME "INT8 penuh"
#else
#include "phasenet_3ch_16x8_model.h"
#include "hw_eval_3ch_100.h"
typedef int16_t qout_t;
static const qout_t QOUT_MIN = -32768;
#define QOUT_NAME "int16"
#define VARIAN_NAME "16x8"
#endif

Adafruit_INA219 ina219;
static bool g_has_ina219 = false;

static uint8_t* tensor_arena = nullptr;
static size_t tensor_arena_size = 0;
static const char* arena_location = "Unknown";

static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

constexpr int WARMUP_REPS = 5;
constexpr int MEASURE_REPS = HW_3CH_N; // Exactly 100 test windows (50 earthquake + 50 noise)
constexpr int IDLE_MS = 2000;

static float invoke_ms[MEASURE_REPS];

// --- Sampler daya (core 0) ---
enum Phase : int { PHASE_NONE = 0, PHASE_IDLE = 1, PHASE_ACTIVE = 2 };
static volatile int g_phase = PHASE_NONE;
static volatile bool g_sampler_run = true;
static volatile double g_sum_mW[3] = {0, 0, 0};
static volatile uint32_t g_n[3] = {0, 0, 0};

void powerSamplerTask(void*) {
  while (g_sampler_run) {
    if (g_has_ina219) {
      int before = g_phase;
      float p = ina219.getPower_mW();
      int after = g_phase;
      if (before == after && before != PHASE_NONE) {
        g_sum_mW[before] += p;
        g_n[before]++;
      }
    }
    vTaskDelay(1); // FreeRTOS 1000 Hz
  }
  vTaskDelete(NULL);
}

void computeStats(const float* d, int n, float* mean, float* sd, float* mn, float* mx, float* med) {
  double s = 0, s2 = 0;
  *mn = d[0]; *mx = d[0];
  std::vector<float> sorted(d, d + n);
  std::sort(sorted.begin(), sorted.end());
  *med = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0f : sorted[n / 2];

  for (int i = 0; i < n; i++) {
    s += d[i];
    s2 += (double)d[i] * d[i];
    if (d[i] < *mn) *mn = d[i];
    if (d[i] > *mx) *mx = d[i];
  }
  double m = s / n;
  double v = s2 / n - m * m;
  *mean = (float)m;
  *sd = (float)(v > 0 ? sqrt(v) : 0.0);
}

// Maximum P probability over 3072 time steps
// Output shape: [1, 3072, 3] = [P, S, Noise]
float maxPScore() {
  const int steps = model_output->dims->data[1];
  const int channels = model_output->dims->data[2];
  const float sc = model_output->params.scale;
  const int zp = model_output->params.zero_point;
  const qout_t* o = reinterpret_cast<const qout_t*>(model_output->data.raw);

  qout_t best = QOUT_MIN;
  for (int t = 0; t < steps; t++) {
    qout_t v = o[t * channels + 0]; // Channel 0 = P-wave
    if (v > best) best = v;
  }
  return (best - zp) * sc;
}

void profilingTask(void*) {
  Serial.println("[TASK] Memulai rangkaian profiling PhaseNet 3-Kanal 16x8...");

  if (g_has_ina219) {
    g_phase = PHASE_IDLE;
    vTaskDelay(pdMS_TO_TICKS(IDLE_MS));
    g_phase = PHASE_NONE;
  }

  uint32_t tp = 0, tn = 0, fp = 0, fn = 0;
  float max_abs_diff = 0.0f;
  int mismatched_decisions = 0;

  Serial.println("[INFO] Menjalankan pemanasan & iterasi pengukuran...");
  for (int r = -WARMUP_REPS; r < MEASURE_REPS; r++) {
    const int w = (r + WARMUP_REPS) % HW_3CH_N;

    // Load the 3-channel data into the input tensor
    memcpy(model_input->data.raw, HW_3CH_INPUTS[w], sizeof(HW_3CH_INPUTS[w]));

    if (r >= 0 && g_has_ina219) g_phase = PHASE_ACTIVE;

    int64_t t0 = esp_timer_get_time();
    TfLiteStatus st = interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();

    if (r >= 0 && g_has_ina219) g_phase = PHASE_NONE;

    if (st != kTfLiteOk) {
      Serial.println("[ERROR] interpreter->Invoke() gagal!");
      break;
    }

    float elapsed_ms = (t1 - t0) / 1000.0f;
    float score = maxPScore();
#ifdef PRINT_HASH
    if (r >= 0 && r < HW_3CH_N) { TfLiteTensor* o__ = interpreter->output(0); uint32_t h__ = 2166136261u; const uint8_t* d__ = (const uint8_t*)o__->data.raw; for (size_t i__ = 0; i__ < o__->bytes; i__++) { h__ ^= d__[i__]; h__ *= 16777619u; } Serial.printf("HASH %d %08x\n", w, (unsigned)h__); }
#endif

    if (r >= 0) {
      invoke_ms[r] = elapsed_ms;

      const bool pred = score >= HW_3CH_THRESHOLD;
      const bool label = HW_3CH_LABELS[w] == 1;

      if (r < HW_3CH_N) { // Pass 1 for the detection metrics
        if (pred && label) tp++;
        else if (!pred && !label) tn++;
        else if (pred && !label) fp++;
        else fn++;
      }

      float diff = fabsf(score - HW_3CH_PY_SCORES[w]);
      if (diff > max_abs_diff) max_abs_diff = diff;
      if (pred != (HW_3CH_PY_SCORES[w] >= HW_3CH_THRESHOLD)) mismatched_decisions++;

      if ((r + 1) % 10 == 0 || r == 0) {
        Serial.printf("[PROG] Repetisi %2d/%2d: Latensi = %7.2f ms | Skor P = %.5f (Py = %.5f, selisih = %.6f)\n",
                      r + 1, MEASURE_REPS, elapsed_ms, score, HW_3CH_PY_SCORES[w], diff);
      }
    } else {
      Serial.printf("[WARM] Warmup %d: Latensi = %7.2f ms | Skor P = %.5f (Py = %.5f)\n",
                    r + WARMUP_REPS + 1, elapsed_ms, score, HW_3CH_PY_SCORES[w]);
    }

    vTaskDelay(pdMS_TO_TICKS(10));
  }

  g_sampler_run = false;

  float im, isd, imn, imx, imed;
  computeStats(invoke_ms, MEASURE_REPS, &im, &isd, &imn, &imx, &imed);

  const double idle_mW = (g_has_ina219 && g_n[PHASE_IDLE]) ? g_sum_mW[PHASE_IDLE] / g_n[PHASE_IDLE] : 0.0;
  const double active_mW = (g_has_ina219 && g_n[PHASE_ACTIVE]) ? g_sum_mW[PHASE_ACTIVE] / g_n[PHASE_ACTIVE] : 0.0;

  Serial.println("\n=======================================================");
  Serial.printf(" HASIL PENGUKURAN EMPIRIS PHASENET 3-KANAL (" VARIAN_NAME ")\n");
  Serial.printf(" Model                          | PhaseNet-pndw 16x8 (seed 42)\n");
  Serial.printf(" Input Tensor                   | [1, %d, %d] (" QOUT_NAME ")\n", HW_3CH_WINDOW, HW_3CH_CHANNELS);
  Serial.printf(" Output Tensor                  | [1, %d, %d] (" QOUT_NAME ")\n", HW_3CH_WINDOW, HW_3CH_CHANNELS);
  Serial.println("-------------------------------------------------------");
  Serial.printf(" CPU Frequency                  | %d MHz\n", getCpuFrequencyMhz());
  Serial.printf(" Ukuran Bobot Model (Flash)     | %u byte (%.2f KB)\n", g_model_3ch_16x8_len, g_model_3ch_16x8_len / 1024.0);
  Serial.printf(" Lokasi Tensor Arena            | %s\n", arena_location);
  Serial.printf(" Tensor Arena Digunakan         | %.2f KB (dari alokasi %.2f KB)\n",
                interpreter->arena_used_bytes() / 1024.0, tensor_arena_size / 1024.0);
  Serial.println("-------------------------------------------------------");
  Serial.printf(" Latensi Invoke (Mean +- Std)   | %.2f +- %.2f ms\n", im, isd);
  Serial.printf(" Latensi Invoke (Median)        | %.2f ms\n", imed);
  Serial.printf(" Latensi Invoke (Min - Max)     | %.2f - %.2f ms\n", imn, imx);
  if (g_has_ina219) {
    Serial.printf(" Daya Idle                      | %.2f mW\n", idle_mW);
    Serial.printf(" Daya Aktif (saat Invoke)       | %.2f mW\n", active_mW);
    Serial.printf(" Energi per Inferensi           | %.2f mJ\n", active_mW * im / 1000.0);
    Serial.printf(" Energi Bersih di atas Idle     | %.2f mJ\n", (active_mW - idle_mW) * im / 1000.0);
  } else {
    Serial.printf(" Daya & Energi                  | [INA219 tidak tersambung]\n");
  }
  Serial.println("-------------------------------------------------------");
  Serial.printf(" VALIDASI PORT vs TFLITE PYTHON (N = %d jendela):\n", HW_3CH_N);
  Serial.printf("   Selisih Skor Maksimum        | %.6f\n", max_abs_diff);
  Serial.printf("   Kesesuaian Keputusan         | %d / %d (%.1f%%)\n",
                MEASURE_REPS - mismatched_decisions, MEASURE_REPS,
                100.0f * (MEASURE_REPS - mismatched_decisions) / (float)MEASURE_REPS);
  float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 0;
  float rec = (tp + fn > 0) ? (float)tp / (tp + fn) : 0;
  float f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0;
  Serial.printf(" AKURASI DETEKSI (%d jendela STEAD, ambang %.4f):\n", HW_3CH_N, HW_3CH_THRESHOLD);
  Serial.printf("   TP: %u | TN: %u | FP: %u | FN: %u\n", tp, tn, fp, fn);
  Serial.printf("   Precision: %.4f | Recall: %.4f | F1-Score: %.4f\n", prec, rec, f1);
  Serial.println("=======================================================");
  Serial.println("[INFO] Benchmark selesai. Sistem siap.");
  vTaskDelete(NULL);
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 3000) delay(10);

  Serial.println("\n\n=======================================================");
  Serial.println("   BENCHMARK PHASENET 3-KANAL 16x8 PADA ESP32-S3");
  Serial.println("=======================================================");

  // Cek sensor arus/tegangan INA219 (opsional)
  Wire.begin(17, 18);
  Wire.setClock(400000);
  if (ina219.begin()) {
    g_has_ina219 = true;
    Serial.println("[INFO] Sensor INA219 terdeteksi di I2C (17/18). Telemetri daya AKTIF.");
  } else {
    g_has_ina219 = false;
    Serial.println("[INFO] INA219 tidak terdeteksi. Melanjutkan benchmark latensi murni.");
  }

  // Model schema check
  tflite_model = tflite::GetModel(g_model_3ch_16x8);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Versi skema TFLite model tidak cocok!");
    while (1) delay(100);
  }
  Serial.println("[INFO] Skema TFLite valid.");

  // Full Op Resolver for the depthwise separable PhaseNet 16x8
  static tflite::MicroMutableOpResolver<14> resolver;
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddTransposeConv(tflite::Register_TRANSPOSE_CONV_ACT_FIX());
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddSqueeze();
  resolver.AddConcatenation();
  resolver.AddSoftmax();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddRelu();
  resolver.AddStridedSlice();

  // Periksa memori Internal SRAM vs SPIRAM (PSRAM)
  size_t free_internal = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  size_t free_spiram = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  Serial.printf("[MEM] Blok terbesar Internal SRAM : %u byte (%.2f KB)\n", free_internal, free_internal / 1024.0);
  Serial.printf("[MEM] Blok terbesar SPIRAM (PSRAM): %u byte (%.2f KB)\n", free_spiram, free_spiram / 1024.0);

  // Allocation strategy: if SPIRAM is enabled (> 500 KB), allocate a 1 MB arena from PSRAM
  // Otherwise, try to allocate as much as possible from internal SRAM
#ifdef ARENA_INTERNAL   // uji arena di SRAM internal walau PSRAM tersedia
  constexpr bool kForceInternal = true;
#else
  constexpr bool kForceInternal = false;
#endif
  if (free_spiram > 500 * 1024 && !kForceInternal) {
    tensor_arena_size = 1024 * 1024; // 1 MB di PSRAM
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    arena_location = "SPIRAM (PSRAM 8MB, Octal-SPI @ 80MHz)";
    Serial.println("[MEM] Mengalokasikan 1024 KB Tensor Arena di SPIRAM (PSRAM).");
  } else {
    constexpr size_t kHeapReserve = 32 * 1024;
    tensor_arena_size = (free_internal > kHeapReserve) ? (free_internal - kHeapReserve) & ~(size_t)15 : 0;
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, tensor_arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    arena_location = "Internal SRAM";
    Serial.printf("[MEM] Mengalokasikan %.2f KB Tensor Arena di Internal SRAM.\n", tensor_arena_size / 1024.0);
  }

  if (tensor_arena == nullptr) {
    Serial.println("[ERROR] Gagal alokasi tensor arena!");
    while (1) delay(100);
  }

  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, tensor_arena_size);
  interpreter = &static_interpreter;

  Serial.println("[INFO] Memanggil interpreter->AllocateTensors()...");
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() GAGAL! Memori tensor arena tidak mencukupi.");
    while (1) delay(100);
  }

  size_t used = interpreter->arena_used_bytes();
  Serial.printf("[MEM] AllocateTensors() SUKSES! Tensor arena terpakai: %u byte (%.2f KB)\n",
                used, used / 1024.0);

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.printf("[TENSOR] Input: shape [%d, %d, %d], type %d, scale %f, zp %d\n",
                model_input->dims->data[0], model_input->dims->data[1], model_input->dims->data[2],
                model_input->type, model_input->params.scale, model_input->params.zero_point);
  Serial.printf("[TENSOR] Output: shape [%d, %d, %d], type %d, scale %f, zp %d\n",
                model_output->dims->data[0], model_output->dims->data[1], model_output->dims->data[2],
                model_output->type, model_output->params.scale, model_output->params.zero_point);

  // Buat FreeRTOS Task
  if (g_has_ina219) {
    xTaskCreatePinnedToCore(powerSamplerTask, "PowerSampler", 4096, NULL, 2, NULL, 0);
  }
  xTaskCreatePinnedToCore(profilingTask, "Profiling3Ch", 16384, NULL, 1, NULL, 1);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}

extern "C" void app_main(void) {
  initArduino();
  setup();
  while (true) loop();
}

#endif // MODEL_3CH_16X8

