#ifndef SCANNER_CORE_H
#define SCANNER_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared decode core for web (WASM), iOS, and Android.
 * Algorithms over a float tensor. No weights and no Ultralytics.
 *
 * scanner_decode writes scanner_box: class_id, score, xyxy in source pixels.
 *
 * SCANNER_LAYOUT_YOLOV8 is [1, 4+classes, anchors], channel-major.
 *   cx, cy, w, h, then one probability per class (already 0..1).
 *   Boxes are model-input pixels. A letterbox maps (box - pad) / scale.
 *   A null letterbox stretches box / input_size onto the source image.
 * SCANNER_LAYOUT_YOLOX is [1, anchors, 5+classes].
 *   cx, cy, w, h, objectness, class probabilities.
 *   Score is objectness times the class probability.
 *
 * SCANNER_TASK_VEHICLE keeps COCO car, motorcycle, bus, truck (2, 3, 5, 7).
 *   After NMS it returns one box: the large vehicle nearest the image center.
 *   Confidence does not choose it.
 * SCANNER_TASK_PLATE returns every plate after NMS (at most 20).
 *   When vehicle is non-NULL, a plate is kept only if its center lies on that
 *   box. A NULL vehicle is plate-only detection and still returns plate boxes.
 *
 * NMS matches the JS pipeline: higher score first, drop IoU >= 0.45 against
 * any earlier box, then keep at most SCANNER_MAX_DETECTIONS.
 *
 * Returns SCANNER_OK, or SCANNER_ERR_ARGUMENT / SCANNER_ERR_SHAPE /
 * SCANNER_ERR_CAPACITY. *out_count is the full result size. When that exceeds
 * capacity the first capacity boxes are still written.
 */

#define SCANNER_LAYOUT_YOLOV8 ((int32_t)0)
#define SCANNER_LAYOUT_YOLOX ((int32_t)1)

#define SCANNER_TASK_VEHICLE ((int32_t)0)
#define SCANNER_TASK_PLATE ((int32_t)1)

#define SCANNER_OK ((int32_t)0)
#define SCANNER_ERR_ARGUMENT ((int32_t)1)
#define SCANNER_ERR_SHAPE ((int32_t)2)
#define SCANNER_ERR_CAPACITY ((int32_t)3)

#define SCANNER_MAX_DETECTIONS ((int32_t)20)

typedef struct scanner_box {
  int32_t class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
} scanner_box;

typedef struct scanner_letterbox {
  float scale;
  float pad_x;
  float pad_y;
} scanner_letterbox;

#if defined(_WIN32)
#  define SCANNER_API __declspec(dllexport)
#elif defined(__EMSCRIPTEN__)
#  define SCANNER_API __attribute__((used, visibility("default")))
#else
#  define SCANNER_API __attribute__((visibility("default")))
#endif

SCANNER_API int32_t scanner_decode(
    const float *tensor,
    const int32_t dims[3],
    int32_t layout,
    int32_t task,
    float confidence,
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    const scanner_letterbox *letterbox,
    const scanner_box *vehicle,
    scanner_box *out_boxes,
    int32_t capacity,
    int32_t *out_count);

#ifdef __cplusplus
}
#endif

#endif
