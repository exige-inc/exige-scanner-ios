# Exige Scanner for iOS

Swift package that decodes vehicle and plate tensors and bundles the Ultralytics YOLO11n weights. It does not open a camera, toggle a torch, auto-capture, or check a license key. ONNX Runtime on the device runs the model. Pass that runtime's float output to `ScannerCore.decode`.

The Ultralytics Enterprise license is signed, so this package includes the weights.

## Install

In Xcode: File → Add Package Dependencies, then:

```text
https://github.com/exige-inc/exige-scanner-ios
```

Use the version from tag `0.1.0` (Up to Next Major Version from `0.1.0`). Add the product `ExigeScannerCore` to the app target.

```swift
dependencies: [
    .package(url: "https://github.com/exige-inc/exige-scanner-ios", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "ExigeScannerCore", package: "exige-scanner-ios"),
        ]
    ),
]
```

## Weights

| File | Role | Tensor |
| --- | --- | --- |
| `yolo11n.onnx` | COCO vehicles (classes 2, 3, 5, 7) | `[1, 84, 8400]` |
| `license-plate-finetune-v1n.onnx` | one-class plates | `[1, 5, 8400]` |

Both use layout `ScannerLayout.yolov8`, input 640, letterbox, pad gray 114, RGB divided by 255. Plate text is not in this package. Hashes and the license note are in `ULTRALYTICS-YOLO-NOTICE.txt` next to the files.

```swift
import ExigeScannerCore

let vehicleModel = ScannerWeights.vehicleModelURL
let plateModel = ScannerWeights.plateModelURL
```

Load either URL with ONNX Runtime, then decode:

```swift
let vehicles = try ScannerCore.decode(
    tensor: vehicleTensor,
    dimensions: [1, 84, Int32(anchorCount)],
    layout: .yolov8,
    task: .vehicle,
    confidence: 0.35,
    imageWidth: imageWidth,
    imageHeight: imageHeight,
    inputSize: 640,
    letterbox: ScannerLetterbox(scale: scale, padX: padX, padY: padY)
)
```

`ScannerCore.decode` returns `[ScannerBox]`. A vehicle call returns zero boxes or one box: the large vehicle nearest the image center. Pass that box as `vehicle:` on a plate call to keep a plate only when its center lies on the vehicle. `vehicle: nil` returns every plate box that passes the confidence gate and NMS.

## Build

On a Mac with Xcode, from this repository:

```bash
swift build
ios/build-xcframework.sh
```

`ios/build-xcframework.sh` needs `xcodebuild` and writes `ios/build/ExigeScannerCore.xcframework`.
