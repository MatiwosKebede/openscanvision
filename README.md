## Overview
OpenScanVision is a standalone Android application for scanning printed voting cards (or any bubble-sheet form) using optical mark recognition (OMR) and QR code decoding. It is designed to be accurate, fast, and easy to integrate into election or survey workflows.

The application runs entirely offline – no internet connection is required. It uses CameraX for camera handling, ML Kit for QR reading, OpenCV for perspective correction and contour detection, and Jetpack Compose for the user interface.


## Features
- Real-time preview with a live overlay that shows:
  - Card edges (green when stable)
  - QR code area (magenta highlight)
  - Reference markers (TL, BR, BL) if present
  - Each bubble’s status (blue = empty, green = filled)
- High-accuracy OMR pipeline:
  - Perspective correction via homography
  - Histogram equalisation, median denoising, and normalisation
  - Weighted circular bubble sampling
  - Otsu thresholding with confidence scoring
- QR-based card identification – each card type is identified by a short prefix (e.g., `VX` for candidate ballots, `AGN` for agenda voting)
- Automatic capture when the card is held steady for a configurable number of frames
- Manual capture as a fallback
- Confidence score for each scan – helps operators decide whether to accept or rescan
- Exportable results as JSON (displayed in the app, can be extended to file export)
- Manual entry fallback – if the QR code is not readable, the token can be typed manually
- Modular architecture – easy to extend with new card templates, custom preprocessing, or different output formats


## Quick Start

### Prerequisites
- Android Studio (latest version recommended)
- JDK 11 or higher
- Android SDK with minimum API 24 (Android 7.0)
- OpenCV – the project includes `com.quickbirdstudios:opencv:4.5.3.0` – it will be downloaded automatically by Gradle

### Building from Source
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OpenScanVision.git
   cd OpenScanVision
   ```
2. Open the project in Android Studio.
3. Let Gradle sync all dependencies.
4. Connect a physical device (or start an emulator with camera support).
5. Run the `app` module.

> **Note:** The app works best on a physical device with a good camera. Emulator camera support is limited.


## How It Works

### Card Template Design
Each card type is defined in `CardTemplate.kt` as a `CardTemplate` instance. A template specifies:
- `prefix` – a short string that appears in the QR code (e.g., `VX`, `AGN`)
- `qrRefCorners` – the four corners of the QR area in a canonical coordinate system (units: 0.1 mm)
- `bubblePositions` – (x, y) coordinates of each bubble in the same canonical space
- `bubbleGroups` – a list of `IntRange` defining independent answer groups (e.g., a single race, or multiple agenda rows)
- `markerRefPositions` – optional positions of reference markers (TL, BR, BL) for homography refinement

The app comes with two predefined templates:
- **Candidate** (`VX`): 12 bubbles, one answer group (single choice)
- **Agenda** (`AGN`): 12 bubbles, 4 groups of 3 (Agree/Disagree/Neutral per resolution)

You can add your own templates by extending the `Templates` object.

### Scanning Pipeline
1. **Frame acquisition** – CameraX provides images at 1280x720 resolution; a downscaled copy (800x450) is used for fast edge detection.
2. **Card detection** – Using OpenCV, the app finds the card quadrilateral (Canny edges + contour approximation).
3. **QR reading** – ML Kit decodes the QR code, returning its content and corner coordinates (full resolution).
4. **Homography calculation** – A perspective transform from template space to image space is computed from the QR corners. If reference markers are defined, they are detected near their predicted positions and added to the correspondence set, improving accuracy.
5. **Bubble projection** – The homography maps bubble positions from template coordinates to image coordinates.
6. **Live classification** – `BubbleAnalyzer` samples bubble darkness (bulk pixel reading) and classifies each bubble as filled or empty using a per-group z-score. This drives the overlay in real time.
7. **Stability check** – The app tracks motion of the card corners. When movement falls below `MOTION_EPSILON_PX` for `STABLE_FRAMES_REQUIRED` consecutive frames, auto-capture is triggered.
8. **Final capture** – The last 3–5 full-resolution frames are averaged to reduce noise. The averaged bitmap is then passed to `OMRExtractor`, which:
   - Preprocesses (contrast enhancement, median denoising, normalisation)
   - Warps the image to the canonical template size (850×540)
   - Samples each bubble using a weighted circular window
   - Computes an Otsu threshold across all bubble intensities
   - Returns the indices of bubbles darker than the threshold, along with a confidence score
9. **Result display** – The token, filled indices, confidence, and JSON representation are shown in a dialog.


## Usage
1. Grant camera permission when prompted.
2. Hold the printed card in good light, aligning it within the dashed frame.
3. The overlay will show the card outline (green = stable), QR area, markers (if present), and each bubble (blue = empty, green = filled).
4. Keep the card steady – after about 1 second, auto‑capture will trigger and show the result.
5. Alternatively, tap **Capture & Verify** to manually trigger a scan.
6. The result dialog displays:
   - **Token** – the QR content (card identifier)
   - **Confidence** – a percentage (0–100%)
   - **Filled indices** – which bubbles were marked (zero‑based index, useful for mapping to candidates/resolutions)
   - **JSON** – raw data for export or integration
7. You can **Rescan** or **OK** to dismiss.
8. If QR is not readable, use the **Manual Entry** toggle to type the token.


## Project Structure

```
app/src/main/java/org/openscanvision/
├── MainActivity.kt                 # App entry point
├── OpenScanVisionApplication.kt    # Initialises OpenCV
├── model/
│   └── LocalScanResult.kt          # Data class for scan results
├── omr/                            # OMR Engine (core logic)
│   ├── BubbleAnalyzer.kt           # Fast classification for live overlay
│   ├── CardDetector.kt             # Quadrilateral detection + homography
│   ├── CardTemplate.kt             # Template definitions (CANDIDATE, AGENDA)
│   ├── ImagePreprocessor.kt        # Image enhancement routines
│   ├── ImageProxyExt.kt            # CameraX -> Bitmap conversion
│   └── OMRExtractor.kt             # Accurate OMR pipeline (warp + Otsu)
├── ui/
│   ├── components/
│   │   └── ScannerOverlays.kt      # Live preview overlays
│   ├── screens/
│   │   └── ScannerScreen.kt        # Main scanning screen (camera + state)
│   ├── theme/
│   │   ├── Color.kt                # Brand colours
│   │   ├── Theme.kt                # Dark/light theme + typography
│   │   └── Type.kt                 # (optional) extra text styles
│   └── utils/
│       └── ScannerUtils.kt         # Coordinate mapping, downsampling
└── res/                            # Resources (icons, strings, layouts)
```


## Extending the Project

### Adding a New Card Template
1. Open `omr/CardTemplate.kt`.
2. Define a new `CardTemplate` instance:
   ```kotlin
   val MY_TEMPLATE = CardTemplate(
       name = "My Template",
       prefix = "MY",
       qrRefCorners = SHARED_QR_CORNERS,  // or custom
       bubblePositions = listOf(...),     // your bubble coordinates
       bubbleGroups = listOf(0..5),       // one group with 6 bubbles
       markerRefPositions = SHARED_MARKER_CORNERS // optional
   )
   ```
3. Add it to the `fromPrefix` function in `Templates`:
   ```kotlin
   when {
       prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
       prefix.startsWith(AGENDA.prefix) -> AGENDA
       prefix.startsWith("MY") -> MY_TEMPLATE
       else -> null
   }
   ```

### Tuning Parameters
- `BUBBLE_RADIUS` in `OMRExtractor.kt` – increase if bubbles appear too small; decrease if sampling picks up neighbouring marks.
- `zThreshold` in `BubbleAnalyzer.classifyByGroup` – lower (e.g., 1.0) to detect lighter marks; higher (e.g., 1.8) to be stricter.
- `MOTION_EPSILON_PX` and `STABLE_FRAMES_REQUIRED` in `ScannerScreen.kt` – adjust auto‑capture sensitivity.

### Customising the Output
`LocalScanResult` holds token, filled indices, JSON, and confidence. You can extend this class or add new fields. The result dialog can be modified in `ScannerScreen.kt` where the `AlertDialog` is built.


## Contributing
We welcome contributions of all kinds. Please read our [CONTRIBUTING](./CONTRIBUTING) file for detailed guidelines on code style, testing, and the pull request process.


## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.


## Acknowledgments
- [OpenCV](https://opencv.org/) – for image processing and homography.
- [Google ML Kit](https://developers.google.com/ml-kit) – for QR code scanning.
- [CameraX](https://developer.android.com/training/camerax) – for camera lifecycle management.
- [Jetpack Compose](https://developer.android.com/jetpack/compose) – for modern UI.


## Contact & Support
For issues, questions, or suggestions, please open an [issue on GitHub](https://github.com/matiwoskebed/OpenScanVision/issues).  
For security vulnerabilities, contact us privately at matiwoskebede01@gmail.com.com.
