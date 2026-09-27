import CScannerCore

public struct ScannerBox: Equatable, Sendable {
    public var classId: Int32
    public var score: Float
    public var x1: Float
    public var y1: Float
    public var x2: Float
    public var y2: Float

    public init(classId: Int32, score: Float, x1: Float, y1: Float, x2: Float, y2: Float) {
        self.classId = classId
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }
}

public enum ScannerLayout: Int32, Sendable {
    case yolov8 = 0
    case yolox = 1
}

public enum ScannerTask: Int32, Sendable {
    case vehicle = 0
    case plate = 1
}

public struct ScannerLetterbox: Equatable, Sendable {
    public var scale: Float
    public var padX: Float
    public var padY: Float

    public init(scale: Float, padX: Float, padY: Float) {
        self.scale = scale
        self.padX = padX
        self.padY = padY
    }
}

public enum ScannerCoreError: Error, Equatable {
    case argument
    case shape
    case capacity
    case unknown(Int32)
}

public enum ScannerCore {
    public static let maxDetections = 20

    public static func decode(
        tensor: [Float],
        dimensions: [Int32],
        layout: ScannerLayout,
        task: ScannerTask,
        confidence: Float,
        imageWidth: Int32,
        imageHeight: Int32,
        inputSize: Int32,
        letterbox: ScannerLetterbox? = nil,
        vehicle: ScannerBox? = nil
    ) throws -> [ScannerBox] {
        guard dimensions.count == 3 else { throw ScannerCoreError.argument }
        let expected = Int(dimensions[0]) * Int(dimensions[1]) * Int(dimensions[2])
        guard expected >= 0, tensor.count == expected else { throw ScannerCoreError.argument }

        var dims = dimensions
        var letter = letterbox.map {
            scanner_letterbox(scale: $0.scale, pad_x: $0.padX, pad_y: $0.padY)
        }
        var vehicleBox = vehicle.map {
            scanner_box(
                class_id: $0.classId,
                score: $0.score,
                x1: $0.x1,
                y1: $0.y1,
                x2: $0.x2,
                y2: $0.y2
            )
        }
        var output = [scanner_box](
            repeating: scanner_box(class_id: 0, score: 0, x1: 0, y1: 0, x2: 0, y2: 0),
            count: maxDetections
        )
        var count: Int32 = 0

        let status: Int32 = tensor.withUnsafeBufferPointer { tensorPointer in
            dims.withUnsafeMutableBufferPointer { dimsPointer in
                output.withUnsafeMutableBufferPointer { outputPointer in
                    withOptionalPointer(&letter) { letterPointer in
                        withOptionalPointer(&vehicleBox) { vehiclePointer in
                            scanner_decode(
                                tensorPointer.baseAddress,
                                dimsPointer.baseAddress,
                                layout.rawValue,
                                task.rawValue,
                                confidence,
                                imageWidth,
                                imageHeight,
                                inputSize,
                                letterPointer,
                                vehiclePointer,
                                outputPointer.baseAddress,
                                Int32(maxDetections),
                                &count
                            )
                        }
                    }
                }
            }
        }

        switch status {
        case 0:
            return output.prefix(Int(count)).map {
                ScannerBox(classId: $0.class_id, score: $0.score, x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2)
            }
        case 1:
            throw ScannerCoreError.argument
        case 2:
            throw ScannerCoreError.shape
        case 3:
            throw ScannerCoreError.capacity
        default:
            throw ScannerCoreError.unknown(status)
        }
    }
}

private func withOptionalPointer<T, R>(_ value: inout T?, _ body: (UnsafePointer<T>?) -> R) -> R {
    guard var stored = value else { return body(nil) }
    return withUnsafePointer(to: &stored) { body($0) }
}
