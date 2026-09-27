# Host toolchain file for the iOS static library.
# CMake 3.14+ can target iOS on macOS. This file only records that contract
# and refuses a Linux configure, where the iPhone SDK does not exist.

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR
    "The iOS toolchain requires macOS with Xcode. "
    "This environment cannot compile the iOS library. "
    "Sources stay in native/src; Swift callers use native/Package.swift.")
endif()

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum iOS version")
if(NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "iOS architectures")
endif()
set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
