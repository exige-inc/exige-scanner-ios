import Foundation

/// Ultralytics YOLO11n files bundled in this package.
/// The C++ core still only decodes tensors. ONNX Runtime on the device loads these files.
public enum ScannerWeights {
    public static let vehicleFileName = "yolo11n.onnx"
    public static let plateFileName = "license-plate-finetune-v1n.onnx"

    /// COCO YOLO11n. Output `[1, 84, 8400]`, layout `yolov8`, letterbox 640.
    public static var vehicleModelURL: URL? {
        modelURL(name: "yolo11n")
    }

    /// One-class plate YOLO11n. Output `[1, 5, 8400]`, layout `yolov8`, letterbox 640.
    public static var plateModelURL: URL? {
        modelURL(name: "license-plate-finetune-v1n")
    }

    public static var noticeURL: URL? {
        Bundle.module.url(forResource: "ULTRALYTICS-YOLO-NOTICE", withExtension: "txt", subdirectory: "Models")
    }

    private static func modelURL(name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "onnx", subdirectory: "Models")
    }
}
