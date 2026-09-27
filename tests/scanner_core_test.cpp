#include "include/scanner_core.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void fail(const char *file, int line, const std::string &message) {
  std::fprintf(stderr, "FAIL %s:%d %s\n", file, line, message.c_str());
  g_failed += 1;
}

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) fail(__FILE__, __LINE__, #condition);                              \
  } while (0)

bool near(float actual, double expected) { return std::fabs(static_cast<double>(actual) - expected) <= 1e-4; }

struct Call {
  std::vector<float> tensor;
  int32_t dims[3] = {1, 0, 0};
  int32_t layout = SCANNER_LAYOUT_YOLOV8;
  int32_t task = SCANNER_TASK_VEHICLE;
  float confidence = 0.35f;
  int32_t image_width = 640;
  int32_t image_height = 640;
  int32_t input_size = 640;
  bool use_letterbox = false;
  scanner_letterbox letterbox = {1.f, 0.f, 0.f};
  bool use_vehicle = false;
  scanner_box vehicle = {};
  int32_t status = 0;
  int32_t count = -1;
  std::vector<scanner_box> boxes;
};

void write_yolov8(
    std::vector<float> &tensor,
    int32_t anchors,
    int32_t anchor,
    float center_x,
    float center_y,
    float width,
    float height,
    int32_t class_id,
    float score) {
  tensor[static_cast<size_t>(anchor)] = center_x;
  tensor[static_cast<size_t>(anchors + anchor)] = center_y;
  tensor[static_cast<size_t>(anchors * 2 + anchor)] = width;
  tensor[static_cast<size_t>(anchors * 3 + anchor)] = height;
  tensor[static_cast<size_t>(anchors * (4 + class_id) + anchor)] = score;
}

void write_yolox(
    std::vector<float> &tensor,
    int32_t features,
    int32_t anchor,
    float center_x,
    float center_y,
    float width,
    float height,
    float objectness,
    int32_t class_id,
    float class_score) {
  const size_t base = static_cast<size_t>(anchor * features);
  tensor[base] = center_x;
  tensor[base + 1] = center_y;
  tensor[base + 2] = width;
  tensor[base + 3] = height;
  tensor[base + 4] = objectness;
  tensor[base + 5 + static_cast<size_t>(class_id)] = class_score;
}

Call decode(Call input) {
  input.boxes.assign(8, scanner_box{});
  input.count = -1;
  input.status = scanner_decode(
      input.tensor.empty() ? nullptr : input.tensor.data(),
      input.dims,
      input.layout,
      input.task,
      input.confidence,
      input.image_width,
      input.image_height,
      input.input_size,
      input.use_letterbox ? &input.letterbox : nullptr,
      input.use_vehicle ? &input.vehicle : nullptr,
      input.boxes.data(),
      static_cast<int32_t>(input.boxes.size()),
      &input.count);
  return input;
}

void expect_box(const scanner_box &box, int32_t class_id, double score, double x1, double y1, double x2, double y2) {
  const int before = g_failed;
  CHECK(box.class_id == class_id);
  CHECK(near(box.score, score));
  CHECK(near(box.x1, x1));
  CHECK(near(box.y1, y1));
  CHECK(near(box.x2, x2));
  CHECK(near(box.y2, y2));
  if (g_failed != before) {
    std::fprintf(
        stderr,
        "  box class %d score %.6f xyxy %.3f %.3f %.3f %.3f\n",
        box.class_id,
        box.score,
        box.x1,
        box.y1,
        box.x2,
        box.y2);
  }
}

void test_subject_vehicle() {
  constexpr int32_t anchors = 4;
  constexpr int32_t features = 84;
  constexpr int32_t width = 1920;
  constexpr int32_t height = 1080;
  std::vector<float> tensor(static_cast<size_t>(features * anchors), 0.f);
  write_yolov8(tensor, anchors, 0, 170.f, 490.f, 340.f, 260.f, 2, 0.73f);
  write_yolov8(tensor, anchors, 1, 1010.f, 570.f, 980.f, 700.f, 7, 0.46f);
  write_yolov8(tensor, anchors, 2, 960.f, 540.f, 24.f, 24.f, 2, 0.91f);

  Call input;
  input.tensor = tensor;
  input.dims[0] = 1;
  input.dims[1] = features;
  input.dims[2] = anchors;
  input.layout = SCANNER_LAYOUT_YOLOV8;
  input.task = SCANNER_TASK_VEHICLE;
  input.image_width = width;
  input.image_height = height;
  input.input_size = 640;
  input.use_letterbox = true;
  input.letterbox = {1.f, 0.f, 0.f};

  const Call all = decode(input);
  CHECK(all.status == SCANNER_OK);
  CHECK(all.count == 1);
  expect_box(all.boxes[0], 7, 0.46, 520, 220, 1500, 920);

  write_yolov8(input.tensor, anchors, 2, 0.f, 0.f, 0.f, 0.f, 2, 0.f);
  const Call edge = decode(input);
  CHECK(edge.status == SCANNER_OK);
  CHECK(edge.count == 1);
  expect_box(edge.boxes[0], 7, 0.46, 520, 220, 1500, 920);

  input.tensor.assign(tensor.size(), 0.f);
  write_yolov8(input.tensor, anchors, 1, 1010.f, 570.f, 980.f, 700.f, 7, 0.46f);
  write_yolov8(input.tensor, anchors, 2, 960.f, 540.f, 24.f, 24.f, 2, 0.91f);
  const Call speck = decode(input);
  CHECK(speck.status == SCANNER_OK);
  CHECK(speck.count == 1);
  CHECK(speck.boxes[0].class_id == 7);
  CHECK(near(speck.boxes[0].score, 0.46));
}

void test_plate_on_vehicle() {
  constexpr int32_t anchors = 3;
  constexpr int32_t features = 5;
  std::vector<float> tensor(static_cast<size_t>(features * anchors), 0.f);
  write_yolov8(tensor, anchors, 0, 60.f, 32.f, 80.f, 24.f, 0, 0.95f);
  write_yolov8(tensor, anchors, 1, 280.f, 218.f, 120.f, 36.f, 0, 0.88f);
  write_yolov8(tensor, anchors, 2, 95.f, 75.f, 50.f, 30.f, 0, 0.80f);

  Call input;
  input.tensor = tensor;
  input.dims[0] = 1;
  input.dims[1] = features;
  input.dims[2] = anchors;
  input.layout = SCANNER_LAYOUT_YOLOV8;
  input.task = SCANNER_TASK_PLATE;
  input.image_width = 640;
  input.image_height = 480;
  input.input_size = 640;
  input.use_letterbox = true;
  input.letterbox = {1.f, 0.f, 0.f};
  input.use_vehicle = true;
  input.vehicle = {7, 0.46f, 100.f, 80.f, 500.f, 300.f};

  const Call on_vehicle = decode(input);
  CHECK(on_vehicle.status == SCANNER_OK);
  CHECK(on_vehicle.count == 1);
  expect_box(on_vehicle.boxes[0], 0, 0.88, 220, 200, 340, 236);

  input.use_vehicle = false;
  const Call plates = decode(input);
  CHECK(plates.status == SCANNER_OK);
  CHECK(plates.count == 3);
  CHECK(near(plates.boxes[0].score, 0.95));
  CHECK(near(plates.boxes[0].x1, 20));
}

void test_empty() {
  Call input;
  input.tensor.assign(84 * 4, 0.f);
  input.dims[0] = 1;
  input.dims[1] = 84;
  input.dims[2] = 4;
  input.task = SCANNER_TASK_VEHICLE;
  input.use_letterbox = true;
  input.letterbox = {1.f, 0.f, 0.f};
  input.image_width = 1920;
  input.image_height = 1080;
  const Call zeros = decode(input);
  CHECK(zeros.status == SCANNER_OK);
  CHECK(zeros.count == 0);

  write_yolov8(input.tensor, 4, 0, 320.f, 320.f, 80.f, 80.f, 0, 0.99f);
  const Call person = decode(input);
  CHECK(person.status == SCANNER_OK);
  CHECK(person.count == 0);

  input.tensor.assign(5 * 1, 0.f);
  input.dims[1] = 5;
  input.dims[2] = 1;
  input.task = SCANNER_TASK_PLATE;
  write_yolov8(input.tensor, 1, 0, 60.f, 32.f, 80.f, 24.f, 0, 0.95f);
  input.use_vehicle = true;
  input.vehicle = {7, 0.46f, 400.f, 300.f, 600.f, 420.f};
  const Call missed = decode(input);
  CHECK(missed.status == SCANNER_OK);
  CHECK(missed.count == 0);
}

void test_yolox_letterbox_and_stretch() {
  constexpr int32_t anchors = 2;
  constexpr int32_t features = 85;
  std::vector<float> tensor(static_cast<size_t>(anchors * features), 0.f);
  write_yolox(tensor, features, 0, 40.f, 40.f, 20.f, 20.f, 0.95f, 0, 0.99f);
  write_yolox(tensor, features, 1, 208.f, 208.f, 80.f, 40.f, 0.9f, 2, 0.8f);

  Call input;
  input.tensor = tensor;
  input.dims[0] = 1;
  input.dims[1] = anchors;
  input.dims[2] = features;
  input.layout = SCANNER_LAYOUT_YOLOX;
  input.task = SCANNER_TASK_VEHICLE;
  input.image_width = 832;
  input.image_height = 416;
  input.input_size = 416;
  input.use_letterbox = true;
  input.letterbox = {0.5f, 0.f, 104.f};
  const Call car = decode(input);
  CHECK(car.status == SCANNER_OK);
  CHECK(car.count == 1);
  expect_box(car.boxes[0], 2, 0.72, 336, 168, 496, 248);

  constexpr int32_t plate_anchors = 8;
  constexpr int32_t plate_features = 6;
  std::vector<float> plates(static_cast<size_t>(plate_anchors * plate_features), 0.f);
  write_yolox(plates, plate_features, 0, 320.f, 400.f, 160.f, 40.f, 0.9f, 0, 0.5f);
  Call plate;
  plate.tensor = plates;
  plate.dims[0] = 1;
  plate.dims[1] = plate_anchors;
  plate.dims[2] = plate_features;
  plate.layout = SCANNER_LAYOUT_YOLOX;
  plate.task = SCANNER_TASK_PLATE;
  plate.image_width = 1280;
  plate.image_height = 640;
  plate.input_size = 640;
  const Call stretched = decode(plate);
  CHECK(stretched.status == SCANNER_OK);
  CHECK(stretched.count == 1);
  expect_box(stretched.boxes[0], 0, 0.45, 480, 380, 800, 420);
}

void test_channel_major_truck() {
  constexpr int32_t anchors = 3;
  constexpr int32_t features = 12;
  std::vector<float> tensor(static_cast<size_t>(features * anchors), 0.f);
  write_yolov8(tensor, anchors, 0, 100.f, 80.f, 40.f, 20.f, 7, 0.66f);
  Call input;
  input.tensor = tensor;
  input.dims[0] = 1;
  input.dims[1] = features;
  input.dims[2] = anchors;
  input.task = SCANNER_TASK_VEHICLE;
  const Call truck = decode(input);
  CHECK(truck.status == SCANNER_OK);
  CHECK(truck.count == 1);
  expect_box(truck.boxes[0], 7, 0.66, 80, 70, 120, 90);
}

void test_plate_nms_drops_overlap() {
  constexpr int32_t anchors = 3;
  constexpr int32_t features = 5;
  std::vector<float> tensor(static_cast<size_t>(features * anchors), 0.f);
  write_yolov8(tensor, anchors, 0, 100.f, 100.f, 40.f, 20.f, 0, 0.90f);
  write_yolov8(tensor, anchors, 1, 110.f, 100.f, 40.f, 20.f, 0, 0.60f);
  write_yolov8(tensor, anchors, 2, 400.f, 100.f, 40.f, 20.f, 0, 0.50f);
  Call input;
  input.tensor = tensor;
  input.dims[0] = 1;
  input.dims[1] = features;
  input.dims[2] = anchors;
  input.task = SCANNER_TASK_PLATE;
  input.use_letterbox = true;
  input.letterbox = {1.f, 0.f, 0.f};
  const Call plates = decode(input);
  CHECK(plates.status == SCANNER_OK);
  CHECK(plates.count == 2);
  CHECK(near(plates.boxes[0].score, 0.90));
  CHECK(near(plates.boxes[0].x1, 80));
  CHECK(near(plates.boxes[1].score, 0.50));
  CHECK(near(plates.boxes[1].x1, 380));
}

void test_rejects_bad_shape() {
  int32_t count = 7;
  const int32_t dims[3] = {1, 4, 2};
  const float tensor[8] = {};
  const int32_t status = scanner_decode(
      tensor, dims, SCANNER_LAYOUT_YOLOV8, SCANNER_TASK_VEHICLE, 0.35f, 640, 640, 640, nullptr, nullptr, nullptr, 0, &count);
  CHECK(status == SCANNER_ERR_SHAPE);
  CHECK(count == 0);
}

}  // namespace

int main() {
  test_subject_vehicle();
  test_plate_on_vehicle();
  test_empty();
  test_yolox_letterbox_and_stretch();
  test_channel_major_truck();
  test_plate_nms_drops_overlap();
  test_rejects_bad_shape();
  if (g_failed) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("scanner_core_tests passed\n");
  return 0;
}
