OpenScanVision – OMR + QR Scanner for Android

OpenScanVision is a standalone Android application that scans printed voting cards or any bubble‑sheet form using optical mark recognition (OMR) and QR code decoding. It is designed to be accurate, fast, and easy to integrate into election or survey workflows.

Features:
- Real‑time preview with live overlay showing card edges, QR corners, reference markers, and bubble status (filled/empty).
- High‑accuracy OMR via perspective correction, histogram equalisation, median filtering, and Otsu thresholding.
- QR‑based card identification – each card type is identified by a short prefix, e.g., VX for candidate ballots, AGN for agenda voting.
- Automatic capture when the card is held steady – no button press required.
- Manual capture as a fallback.
- Confidence scoring for each scan.
- Exportable results as JSON.
- Fully offline – no internet connection needed.
- Modular architecture – easy to extend with new templates or custom preprocessing.

Getting Started:

Prerequisites:
- Android Studio (latest version recommended)
- JDK 11 or higher
- Android SDK with minimum API level 24 (Android 7.0)
- OpenCV – the project includes com.quickbirdstudios:opencv:4.5.3.0 – it will be downloaded automatically.

Building from Source:
1. Clone the repository:
   git clone https://github.com/yourusername/OpenScanVision.git
   cd OpenScanVision
2. Open the project in Android Studio.
3. Let Gradle sync all dependencies.
4. Connect a physical device or start an emulator with camera support.
5. Run the app module.

The app requires a camera; it works best on a real device with a good camera sensor.

How It Works:

Card Template Design:
Each card type is defined in CardTemplate.kt. A template specifies:
- prefix – a short string that appears in the QR code (e.g., VX, AGN).
- qrRefCorners – the four corners of the QR area in a canonical coordinate system (0.1 mm units).
- bubblePositions – (x,y) coordinates of each bubble in the same system.
- bubbleGroups – a list of index ranges defining independent answer groups (e.g., one race, or multiple agenda rows).
- markerRefPositions – optional positions of reference markers (TL, BR, BL) for homography refinement.

The app comes with two predefined templates:
- Candidate (VX): 12 bubbles, one answer group (single choice).
- Agenda (AGN): 12 bubbles, 4 groups of 3 (Agree/Disagree/Neutral per resolution).

You can add your own templates by extending the Templates object.

Scanning Pipeline:
1. Camera frame is converted to Bitmap (full resolution for QR/markers, downsampled for speed).
2. Card quadrilateral is found via OpenCV (Canny + contours) – used for visual overlay.
3. QR code is decoded by ML Kit – provides content and corner coordinates.
4. Homography is computed from QR corners to template reference, optionally refined with detected reference markers.
5. Bubble positions are mapped from template to image space using the homography.
6. Live classification (BubbleAnalyzer) samples each bubble and marks them as filled/empty for the overlay.
7. Stability check: if the card is steady for a configurable number of frames, auto‑capture triggers.
8. Final capture: averages the last 3–5 frames, then OMRExtractor:
   - Preprocesses (contrast, denoise, normalise)
   - Warps the image to canonical template size
   - Samples bubbles (weighted circular)
   - Applies Otsu thresholding
   - Returns filled indices and confidence

Usage:
1. Grant camera permission when prompted.
2. Hold the printed card in good light, aligning it within the dashed frame.
3. The overlay will show the card outline, QR area, markers, and bubbles.
4. Keep the card steady – after about 1 second, auto‑capture will fire and show the result.
5. Alternatively, tap Capture & Verify to manually trigger a scan.
6. The result dialog displays:
   - Token – the QR content
   - Confidence – a percentage (0–100%)
   - Filled indices – which bubbles were marked (zero‑based index)
   - JSON – raw data for export
7. You can rescan or OK to dismiss.
8. If QR cannot be read, use the Manual Entry toggle to type the token.

Project Structure:

app/src/main/java/org/openscanvision/
├── MainActivity.kt                 # App entry
├── OpenScanVisionApplication.kt    # Initialises OpenCV
├── model/
│   └── LocalScanResult.kt          # Scan result data class
├── omr/                            # OMR Engine
│   ├── BubbleAnalyzer.kt           # Fast classification for overlay
│   ├── CardDetector.kt             # Quadrilateral + homography
│   ├── CardTemplate.kt             # Template definitions
│   ├── ImagePreprocessor.kt        # Image enhancement
│   ├── ImageProxyExt.kt            # CameraX -> Bitmap conversion
│   └── OMRExtractor.kt             # Accurate OMR pipeline
├── ui/
│   ├── components/
│   │   └── ScannerOverlays.kt      # Live preview overlays
│   ├── screens/
│   │   └── ScannerScreen.kt        # Main scanning screen
│   ├── theme/
│   │   ├── Color.kt                # Brand colours
│   │   ├── Theme.kt                # Dark/light theme + typography
│   │   └── Type.kt                 # (optional) extra styles
│   └── utils/
│       └── ScannerUtils.kt         # Coordinate mapping, downsampling
└── res/                            # Resources (icons, strings, etc.)

Extending the Project:

Adding a New Card Template:
1. Open omr/CardTemplate.kt.
2. Define a new CardTemplate instance:
   val MY_TEMPLATE = CardTemplate(
       name = "My Template",
       prefix = "MY",
       qrRefCorners = SHARED_QR_CORNERS,  // or custom
       bubblePositions = listOf(...),     // your bubble coordinates
       bubbleGroups = listOf(0..5),       // one group with 6 bubbles
       markerRefPositions = SHARED_MARKER_CORNERS // optional
   )
3. Add it to the fromPrefix function in Templates:
   when {
       prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
       prefix.startsWith(AGENDA.prefix) -> AGENDA
       prefix.startsWith("MY") -> MY_TEMPLATE
       else -> null
   }

Tuning Parameters:
- BUBBLE_RADIUS in OMRExtractor.kt – increase if bubbles appear too small.
- zThreshold in BubbleAnalyzer.classifyByGroup – lower to detect lighter marks, higher to be stricter.
- MOTION_EPSILON_PX and STABLE_FRAMES_REQUIRED in ScannerScreen.kt – adjust auto‑capture sensitivity.

Customising the Output:
LocalScanResult holds token, filled indices, JSON, and confidence. You can extend this class or add new fields. The result dialog can be modified in ScannerScreen.kt where the AlertDialog is built.

Contributing:
We welcome contributions! Please read our CONTRIBUTING file for details on code style, testing, and pull request process.

License:
This project is licensed under the MIT License – see the LICENSE file for details.

Acknowledgments:
- OpenCV – for image processing and homography.
- Google ML Kit – for QR code scanning.
- CameraX – for camera lifecycle management.
- Jetpack Compose – for modern UI.

Contact & Support:
For issues, questions, or suggestions, please open an issue on GitHub. For security vulnerabilities, please contact us privately at security@example.com.

Happy scanning!

===============================================================================

LICENSE

MIT License

Copyright (c) 2026 OpenScanVision Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

===============================================================================

CONTRIBUTING

Thank you for your interest in contributing to OpenScanVision! We welcome all kinds of contributions – bug reports, feature requests, documentation improvements, and code changes.

How to Contribute:
1. Fork the repository.
2. Create a new branch for your feature or fix:
   git checkout -b feature/your-feature-name
3. Make your changes following the code style and guidelines below.
4. Test your changes thoroughly on a physical device (or emulator with camera).
5. Commit with a clear and descriptive message.
6. Push to your fork and open a Pull Request against the main branch.

Code Style:
- Kotlin: Use the official Kotlin coding conventions (IntelliJ default).
- Compose: Prefer @Composable functions over classes; keep UI logic separate from business logic.
- Naming: Use clear, descriptive names for variables, functions, and classes.
- Comments: Add KDoc comments for public APIs and explain non‑obvious logic.

Testing:
- Unit tests for utility functions (e.g., ImagePreprocessor, ScannerUtils) are encouraged.
- Manual testing with printed cards is essential for OMR changes.
- If you add a new card template, include a sample QR code and test it.

Reporting Issues:
- Use the issue tracker on GitHub.
- Provide a clear description, steps to reproduce, device model, Android version, and if possible, a screenshot or logcat output.

Feature Requests:
- Open an issue with the label 'enhancement'.
- Describe the use case and why it would benefit the project.

Pull Request Checklist:
Before submitting your PR, ensure:
- Code compiles and runs without errors.
- All existing features still work (no regressions).
- New features are documented in the appropriate files (README, code comments).
- No unnecessary debug logs or commented code remain.
- Your commit history is clean (squash if needed).

License:
By contributing, you agree that your contributions will be licensed under the MIT License.

Contact:
If you have any questions, feel free to open a discussion or reach out via the issue tracker.

We appreciate your help in making OpenScanVision better!
