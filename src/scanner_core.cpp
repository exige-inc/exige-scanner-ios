#include "include/scanner_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct RawBox {
  int32_t class_id = 0;
  double score = 0;
  double x1 = 0;
  double y1 = 0;
  double x2 = 0;
  double y2 = 0;
};

bool is_vehicle_class(int32_t class_id) {
  return class_id == 2 || class_id == 3 || class_id == 5 || class_id == 7;
}

double box_width(const RawBox &box) { return box.x2 - box.x1; }

double box_height(const RawBox &box) { return box.y2 - box.y1; }

double overlap(const RawBox &a, const RawBox &b) {
  const double intersection_w = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const double intersection_h = std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const double intersection = intersection_w * intersection_h;
  const double union_area = box_width(a) * box_height(a) + box_width(b) * box_height(b) - intersection;
  return intersection / std::max(1.0, union_area);
}

double box_area(const RawBox &box) {
  return std::max(0.0, box_width(box)) * std::max(0.0, box_height(box));
}

double center_offset(const RawBox &box, double image_width, double image_height) {
  const double width = std::max(1.0, image_width);
  const double height = std::max(1.0, image_height);
  const double dx = ((box.x1 + box.x2) / 2.0 - width / 2.0) / (width / 2.0);
  const double dy = ((box.y1 + box.y2) / 2.0 - height / 2.0) / (height / 2.0);
  return std::hypot(dx, dy);
}

double subject_score(const RawBox &box, double image_width, double image_height) {
  const double width = std::max(1.0, image_width);
  const double height = std::max(1.0, image_height);
  return (box_area(box) / (width * height)) / (1.0 + center_offset(box, image_width, image_height));
}

RawBox map_box(
    double center_x,
    double center_y,
    double width,
    double height,
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    const scanner_letterbox *letterbox) {
  const double left = center_x - width / 2.0;
  const double top = center_y - height / 2.0;
  RawBox box;
  if (letterbox) {
    const double scale = letterbox->scale;
    box.x1 = (left - letterbox->pad_x) / scale;
    box.y1 = (top - letterbox->pad_y) / scale;
    box.x2 = box.x1 + width / scale;
    box.y2 = box.y1 + height / scale;
    return box;
  }
  const double mapped_x = (left / static_cast<double>(input_size)) * static_cast<double>(image_width);
  const double mapped_y = (top / static_cast<double>(input_size)) * static_cast<double>(image_height);
  const double mapped_w = (width / static_cast<double>(input_size)) * static_cast<double>(image_width);
  const double mapped_h = (height / static_cast<double>(input_size)) * static_cast<double>(image_height);
  box.x1 = mapped_x;
  box.y1 = mapped_y;
  box.x2 = mapped_x + mapped_w;
  box.y2 = mapped_y + mapped_h;
  return box;
}

std::vector<RawBox> suppress(std::vector<RawBox> boxes) {
  std::stable_sort(boxes.begin(), boxes.end(), [](const RawBox &a, const RawBox &b) {
    return a.score > b.score;
  });
  std::vector<RawBox> kept;
  kept.reserve(std::min<size_t>(boxes.size(), static_cast<size_t>(SCANNER_MAX_DETECTIONS)));
  for (size_t index = 0; index < boxes.size(); ++index) {
    bool clear = true;
    for (size_t earlier = 0; earlier < index; ++earlier) {
      if (!(overlap(boxes[index], boxes[earlier]) < 0.45)) {
        clear = false;
        break;
      }
    }
    if (!clear) continue;
    kept.push_back(boxes[index]);
    if (kept.size() == static_cast<size_t>(SCANNER_MAX_DETECTIONS)) break;
  }
  return kept;
}

RawBox select_subject(const std::vector<RawBox> &boxes, int32_t image_width, int32_t image_height) {
  RawBox best = boxes[0];
  double best_score = subject_score(best, image_width, image_height);
  double best_area = box_area(best);
  double best_offset = center_offset(best, image_width, image_height);
  for (size_t index = 1; index < boxes.size(); ++index) {
    const RawBox &candidate = boxes[index];
    const double score = subject_score(candidate, image_width, image_height);
    const double area = box_area(candidate);
    const double offset = center_offset(candidate, image_width, image_height);
    const bool preferred = score > best_score || (score == best_score && area > best_area) ||
        (score == best_score && area == best_area && offset < best_offset);
    if (!preferred) continue;
    best = candidate;
    best_score = score;
    best_area = area;
    best_offset = offset;
  }
  return best;
}

bool plate_lies_on_vehicle(const RawBox &plate, const scanner_box &vehicle) {
  const double center_x = (plate.x1 + plate.x2) / 2.0;
  const double center_y = (plate.y1 + plate.y2) / 2.0;
  return center_x >= vehicle.x1 && center_x <= vehicle.x2 && center_y >= vehicle.y1 && center_y <= vehicle.y2;
}

scanner_box to_box(const RawBox &raw) {
  scanner_box box;
  box.class_id = raw.class_id;
  box.score = static_cast<float>(raw.score);
  box.x1 = static_cast<float>(raw.x1);
  box.y1 = static_cast<float>(raw.y1);
  box.x2 = static_cast<float>(raw.x2);
  box.y2 = static_cast<float>(raw.y2);
  return box;
}

}  // namespace

extern "C" SCANNER_API int32_t scanner_decode(
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
    int32_t *out_count) {
  if (!out_count) return SCANNER_ERR_ARGUMENT;
  *out_count = 0;
  if (!dims || capacity < 0 || (capacity > 0 && !out_boxes)) return SCANNER_ERR_ARGUMENT;
  if (image_width <= 0 || image_height <= 0 || input_size <= 0) return SCANNER_ERR_ARGUMENT;
  if (layout != SCANNER_LAYOUT_YOLOV8 && layout != SCANNER_LAYOUT_YOLOX) return SCANNER_ERR_ARGUMENT;
  if (task != SCANNER_TASK_VEHICLE && task != SCANNER_TASK_PLATE) return SCANNER_ERR_ARGUMENT;
  if (!std::isfinite(confidence)) return SCANNER_ERR_ARGUMENT;
  if (letterbox && !(letterbox->scale > 0.f)) return SCANNER_ERR_ARGUMENT;

  const bool yolox = layout == SCANNER_LAYOUT_YOLOX;
  const int32_t batch = dims[0];
  const int32_t anchors = yolox ? dims[1] : dims[2];
  const int32_t features = yolox ? dims[2] : dims[1];
  if (batch != 1 || anchors < 0 || features < 0) return SCANNER_ERR_SHAPE;
  if (features < (yolox ? 6 : 5)) return SCANNER_ERR_SHAPE;
  if (anchors > 0 && features > INT32_MAX / anchors) return SCANNER_ERR_SHAPE;
  if (anchors == 0) return SCANNER_OK;
  if (!tensor) return SCANNER_ERR_ARGUMENT;

  const int32_t class_count = yolox ? features - 5 : features - 4;
  const double minimum = confidence;
  std::vector<RawBox> decoded;
  decoded.reserve(static_cast<size_t>(std::min(anchors, 64)));

  for (int32_t anchor = 0; anchor < anchors; ++anchor) {
    double best_score = 0;
    int32_t best_class = 0;
    double center_x = 0;
    double center_y = 0;
    double width = 0;
    double height = 0;
    if (yolox) {
      const float *row = tensor + static_cast<int64_t>(anchor) * features;
      const double objectness = row[4];
      center_x = row[0];
      center_y = row[1];
      width = row[2];
      height = row[3];
      for (int32_t class_index = 0; class_index < class_count; ++class_index) {
        const double score = objectness * static_cast<double>(row[5 + class_index]);
        if (score > best_score) {
          best_score = score;
          best_class = class_index;
        }
      }
    } else {
      center_x = tensor[anchor];
      center_y = tensor[static_cast<int64_t>(anchors) + anchor];
      width = tensor[static_cast<int64_t>(anchors) * 2 + anchor];
      height = tensor[static_cast<int64_t>(anchors) * 3 + anchor];
      for (int32_t class_index = 0; class_index < class_count; ++class_index) {
        const double score = tensor[static_cast<int64_t>(4 + class_index) * anchors + anchor];
        if (score > best_score) {
          best_score = score;
          best_class = class_index;
        }
      }
    }
    if (best_score < minimum) continue;
    if (task == SCANNER_TASK_VEHICLE && !is_vehicle_class(best_class)) continue;
    RawBox box = map_box(center_x, center_y, width, height, image_width, image_height, input_size, letterbox);
    box.class_id = best_class;
    box.score = best_score;
    decoded.push_back(box);
  }

  std::vector<RawBox> kept = suppress(std::move(decoded));
  if (task == SCANNER_TASK_VEHICLE && kept.size() > 1) {
    kept = {select_subject(kept, image_width, image_height)};
  } else if (task == SCANNER_TASK_PLATE && vehicle) {
    std::vector<RawBox> on_vehicle;
    for (const RawBox &plate : kept) {
      if (plate_lies_on_vehicle(plate, *vehicle)) on_vehicle.push_back(plate);
    }
    kept = std::move(on_vehicle);
  }

  *out_count = static_cast<int32_t>(kept.size());
  const int32_t write = std::min(capacity, *out_count);
  for (int32_t index = 0; index < write; ++index) out_boxes[index] = to_box(kept[static_cast<size_t>(index)]);
  if (*out_count > capacity) return SCANNER_ERR_CAPACITY;
  return SCANNER_OK;
}
