# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] – 2025‑07‑17

### Added
- Full OMR engine with z‑score classification and fill‑ratio analysis.
- ArUco marker tracking (IDs 0–3) with Kalman filter.
- QR code decoding with homography‑based cropping and ML Kit.
- Strict capture logic: 4 markers + valid QR required.
- Sample app with CameraX and Jetpack Compose.
- JitPack publishing support.
- MIT license.
- Comprehensive README, CONTRIBUTING, and CODE_OF_CONDUCT.

### Fixed
- OpenCV initialization stability.
- Camera preview scaling (fit center, no cropping).
- Build and lint errors.
- ImageProxy memory leaks.

### Known Issues
- None reported.


[1.0.0]: https://github.com/MatiwosKebede/OpenScanVision/releases/tag/v1.0.0
