package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Task
import com.google.gson.Gson
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.*
import org.openscanvision.model.LocalScanResult
import org.openscanvision.omr.CardDetector
import org.openscanvision.omr.CardTemplate
import org.openscanvision.omr.OMRExtractor
import org.openscanvision.omr.Templates
import org.openscanvision.ui.components.*
import org.openscanvision.ui.utils.*
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

private const val TAG = "ScannerScreen"
private const val MOTION_EPSILON_PX = 15f
private const val STABLE_FRAMES_REQUIRED = 4

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    // Camera permission
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasCameraPermission = it }

    // UI states
    var isScanning by remember { mutableStateOf(false) }
    var manualSerial by remember { mutableStateOf("") }
    var showManualEntry by remember { mutableStateOf(false) }
    var scanStatus by remember { mutableStateOf("") }
    var scanResult by remember { mutableStateOf<LocalScanResult?>(null) }

    // Live overlay states (for preview) — screen coordinates
    var rawCardCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawQRCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawMarkerCenters by remember { mutableStateOf<Map<String, Offset>?>(null) }
    var qrDetected by remember { mutableStateOf(false) }
    var isTracking by remember { mutableStateOf(false) }

    // Cached data for manual capture
    var latestToken by remember { mutableStateOf<String?>(null) }
    var currentFrameBitmap by remember { mutableStateOf<Bitmap?>(null) }   // raw sensor bitmap, for OMR math only
    var currentQrPoints by remember { mutableStateOf<List<PointF>?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    // Smoothing for overlays (preview only) — these are already in SCREEN coordinates
    val smoothCorners = rememberSmoothCorners(rawCardCorners, smoothing = 0.3f)
    val smoothQRCorners = rememberSmoothCorners(rawQRCorners, smoothing = 0.3f)
    val rawMarkerList = rawMarkerCenters?.values?.toList()
    val smoothMarkerList = rememberSmoothCorners(rawMarkerList, smoothing = 0.3f)
    val smoothMarkerMap = remember(rawMarkerCenters, smoothMarkerList) {
        if (rawMarkerCenters != null && smoothMarkerList != null && rawMarkerCenters!!.size == smoothMarkerList.size) {
            rawMarkerCenters!!.keys.zip(smoothMarkerList).toMap()
        } else {
            null
        }
    }

    val barcodeScanner = remember { BarcodeScanning.getClient() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            barcodeScanner.close()
            currentFrameBitmap?.recycle()
        }
    }

    // ─── Draw overlay directly onto a SCREEN-SPACE snapshot (e.g. previewView.bitmap) ───
    // No rotation/transform math needed here: the coordinates already match what was
    // rendered on screen, and the snapshot itself is already correctly oriented/cropped.
    fun drawScreenOverlayOnBitmap(
        bitmap: Bitmap,
        cardCorners: List<Offset>?,
        qrCorners: List<Offset>?,
        markerCenters: Map<String, Offset>?
    ) {
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }

        // 1. Card outline (green dashed)
        cardCorners?.let { corners ->
            if (corners.size == 4) {
                val path = Path().apply {
                    moveTo(corners[0].x, corners[0].y)
                    lineTo(corners[1].x, corners[1].y)
                    lineTo(corners[2].x, corners[2].y)
                    lineTo(corners[3].x, corners[3].y)
                    close()
                }
                paint.color = Color.Green.toArgb()
                paint.pathEffect = DashPathEffect(floatArrayOf(10f, 8f), 0f)
                canvas.drawPath(path, paint)
                paint.pathEffect = null
            }
        }

        // 2. QR corners (magenta dashed + center dot)
        qrCorners?.let { qrPts ->
            if (qrPts.size == 4) {
                val path = Path().apply {
                    moveTo(qrPts[0].x, qrPts[0].y)
                    lineTo(qrPts[1].x, qrPts[1].y)
                    lineTo(qrPts[2].x, qrPts[2].y)
                    lineTo(qrPts[3].x, qrPts[3].y)
                    close()
                }
                paint.color = Color.Magenta.toArgb()
                paint.pathEffect = DashPathEffect(floatArrayOf(8f, 6f), 0f)
                canvas.drawPath(path, paint)
                paint.pathEffect = null

                // Center dot
                val cx = qrPts.map { it.x }.average().toFloat()
                val cy = qrPts.map { it.y }.average().toFloat()
                paint.style = Paint.Style.FILL
                paint.alpha = 128
                canvas.drawCircle(cx, cy, 12f, paint)
                paint.style = Paint.Style.STROKE
                paint.alpha = 255
            }
        }

        // 3. Marker centers (magenta circles + crosshair)
        markerCenters?.values?.forEach { pt ->
            paint.color = Color.Magenta.toArgb()
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.5f
            // Outer glow
            paint.alpha = 64
            canvas.drawCircle(pt.x, pt.y, 20f, paint)
            paint.alpha = 230
            canvas.drawCircle(pt.x, pt.y, 8f, paint)
            // Cross
            val crossSize = 10f
            canvas.drawLine(pt.x - crossSize, pt.y, pt.x + crossSize, pt.y, paint)
            canvas.drawLine(pt.x, pt.y - crossSize, pt.x, pt.y + crossSize, paint)
            paint.alpha = 255
            // Center dot
            paint.style = Paint.Style.FILL
            canvas.drawCircle(pt.x, pt.y, 4f, paint)
            paint.style = Paint.Style.STROKE
        }
    }

    // ─── Unified Processing Function ────────────────────────────────
    // sensorBitmap: raw ImageAnalysis frame, used ONLY for OMR homography/extraction math.
    // previewSnapshot: previewView.bitmap taken at capture time — exactly what the user saw
    //                  on screen (already rotated/cropped/scaled by CameraX). We draw the
    //                  overlay (already in screen coordinates) directly onto this, so the
    //                  "Original" image shown to the user is a true WYSIWYG capture.
    fun processCurrentFrame(
        sensorBitmap: Bitmap,
        previewSnapshot: Bitmap,
        qrValue: String,
        qrPoints: List<PointF>?,
        cardCornersScreen: List<Offset>?,
        qrCornersScreen: List<Offset>?,
        markerCentersScreen: Map<String, Offset>?
    ) {
        if (isProcessing) return
        isProcessing = true
        isScanning = true
        scanStatus = "Processing image..."

        // Copy the preview snapshot so we don't mutate a bitmap CameraX still owns,
        // then draw the screen-space overlay directly onto it — no rotation needed.
        val displayBitmap = previewSnapshot.copy(
            previewSnapshot.config ?: Bitmap.Config.ARGB_8888, true
        )
        drawScreenOverlayOnBitmap(displayBitmap, cardCornersScreen, qrCornersScreen, markerCentersScreen)

        // ── Process OMR on the raw sensor bitmap (unrelated to display) ──
        coroutineScope.launch(Dispatchers.IO) {
            try {
                val (filledIndices, confidence, warpedBitmap) = OMRExtractor.extractCandidateMarks(
                    originalBitmap = sensorBitmap,
                    qrCorners = qrPoints ?: emptyList()
                )

                withContext(Dispatchers.Main) {
                    scanResult = LocalScanResult(
                        token = qrValue,
                        filledIndices = filledIndices,
                        scanDataJson = Gson().toJson(mapOf(
                            "filled" to filledIndices,
                            "confidence" to confidence
                        )),
                        confidence = confidence,
                        warpedCardBitmap = warpedBitmap,
                        originalBitmap = displayBitmap   // exact preview capture + overlay
                    )
                    isScanning = false
                    isProcessing = false
                    scanStatus = "✅ Captured & Standardised"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Processing error", e)
                withContext(Dispatchers.Main) {
                    scanStatus = "❌ Error: ${e.message}"
                    isScanning = false
                    isProcessing = false
                    displayBitmap.recycle()
                }
            }
        }
    }

    // ─── Camera Setup ────────────────────────────────────────────────
    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()

                val preview = Preview.Builder().build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(1280, 720))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                var lastKnownCorners: List<Offset>? = null
                var lostCount = 0
                val maxLostFrames = 5
                var stableFrameCount = 0
                var lastStableCorners: List<Offset>? = null
                var autoCaptureTriggered = false

                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    if (isProcessing) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val mediaImage = imageProxy.image ?: run { imageProxy.close(); return@setAnalyzer }
                    val bitmap = imageProxy.toBitmap() ?: run { imageProxy.close(); return@setAnalyzer }
                    val imageWidth = imageProxy.width.toFloat()
                    val imageHeight = imageProxy.height.toFloat()
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees

                    val downsampled = downsampleBitmap(bitmap, 800, 450)
                    val processWidth = downsampled.width.toFloat()
                    val processHeight = downsampled.height.toFloat()

                    val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
                    barcodeScanner.process(inputImage)
                        .addOnSuccessListener(cameraExecutor) { barcodes ->
                            val barcode = barcodes.firstOrNull()
                            val qrValue = barcode?.rawValue?.trim()
                            val qrImagePoints = barcode?.cornerPoints?.map { PointF(it.x.toFloat(), it.y.toFloat()) }

                            var finalCardCorners: List<Offset>? = null
                            var finalQRCorners: List<Offset>? = null
                            var markerScreenMap: Map<String, Offset>? = null
                            var template: CardTemplate? = null
                            var imageCorners: List<PointF>? = null
                            var markerCentersImage: List<PointF>? = null

                            // 1. Attempt marker-based detection
                            if (qrValue != null && qrImagePoints != null && qrImagePoints.size == 4) {
                                finalQRCorners = qrImagePoints.map { Offset(it.x, it.y) }
                                template = Templates.fromPrefix(qrValue)

                                if (template != null) {
                                    val homography = OMRExtractor.computeMarkerHomography(bitmap, qrImagePoints, template)
                                    if (homography != null) {
                                        // Card corners in image coordinates
                                        val templateCorners = listOf(
                                            PointF(0f, 0f),
                                            PointF(Templates.REF_WIDTH.toFloat(), 0f),
                                            PointF(Templates.REF_WIDTH.toFloat(), Templates.REF_HEIGHT.toFloat()),
                                            PointF(0f, Templates.REF_HEIGHT.toFloat())
                                        )
                                        imageCorners = CardDetector.predictImagePoints(templateCorners, homography)
                                        // Screen corners for live overlay
                                        val screenCorners = mapImageToScreen(
                                            imageCorners!!.map { Offset(it.x, it.y) },
                                            imageWidth, imageHeight, previewView
                                        )
                                        if (screenCorners != null && screenCorners.size == 4) {
                                            finalCardCorners = screenCorners
                                        }

                                        // ─── Marker centers (image + screen) ─────────────
                                        val markerRefs = template.markerRefPositions ?: Templates.SHARED_MARKER_CORNERS
                                        val predicted = CardDetector.predictImagePoints(markerRefs, homography)
                                        val detected = CardDetector.detectRefMarkersNearPredicted(bitmap, predicted)
                                        val markerLabels = listOf("TL", "TR", "BR", "BL")

                                        // Build lists for image and screen coordinates
                                        val tempMarkerImage = mutableListOf<PointF>()
                                        val tempMarkerScreen = mutableMapOf<String, Offset>()

                                        for (i in markerLabels.indices) {
                                            val pt = detected.getOrNull(i)
                                            if (pt != null) {
                                                tempMarkerImage.add(pt)
                                                // Convert to screen coords for preview overlay
                                                val screenOffset = mapImageToScreen(
                                                    listOf(Offset(pt.x, pt.y)),
                                                    imageWidth, imageHeight, previewView
                                                )?.firstOrNull()
                                                if (screenOffset != null) {
                                                    tempMarkerScreen[markerLabels[i]] = screenOffset
                                                }
                                            }
                                        }

                                        // Only store if we have all 4 markers (optional)
                                        if (tempMarkerImage.size == 4) {
                                            markerCentersImage = tempMarkerImage
                                            markerScreenMap = tempMarkerScreen
                                        }

                                        // Cache data needed for manual capture (sensor bitmap + QR points only —
                                        // display data now comes from previewView.bitmap + smoothed screen state)
                                        coroutineScope.launch(Dispatchers.Main) {
                                            latestToken = qrValue
                                            currentQrPoints = qrImagePoints
                                            currentFrameBitmap?.recycle()
                                            currentFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
                                        }
                                    }
                                }
                            }

                            // 2. Fallback to edge detection (no marker overlay data)
                            if (finalCardCorners == null) {
                                val cardCorners = CardDetector.detectCardCorners(downsampled)
                                if (cardCorners != null && cardCorners.size == 4) {
                                    val scaleX = imageWidth / processWidth
                                    val scaleY = imageHeight / processHeight
                                    finalCardCorners = cardCorners.map { Offset(it.x * scaleX, it.y * scaleY) }
                                }
                            }

                            // Persist corners with lost tracking
                            if (finalCardCorners == null) {
                                lostCount++
                                finalCardCorners = if (lostCount <= maxLostFrames) lastKnownCorners else null
                                if (lostCount > maxLostFrames) lastKnownCorners = null
                            } else {
                                lostCount = 0
                                lastKnownCorners = finalCardCorners
                            }

                            val qrDetectedNow = qrValue != null
                            coroutineScope.launch(Dispatchers.Main) {
                                qrDetected = qrDetectedNow
                                rawMarkerCenters = markerScreenMap
                            }

                            val screenQR = if (finalQRCorners != null) {
                                mapImageToScreen(finalQRCorners, imageWidth, imageHeight, previewView)
                            } else null

                            if (screenQR != null) {
                                coroutineScope.launch(Dispatchers.Main) { rawQRCorners = screenQR }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) { rawQRCorners = null }
                            }

                            if (finalCardCorners != null) {
                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = finalCardCorners
                                    isTracking = true
                                }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) {
                                    isTracking = false
                                    rawCardCorners = null
                                }
                            }

                            // ─── Auto‑capture ──────────────────────────
                            if (finalCardCorners != null && qrValue != null && !autoCaptureTriggered) {
                                val baseline = lastStableCorners
                                val moved = baseline == null || baseline.size != finalCardCorners.size ||
                                        finalCardCorners.indices.any { i ->
                                            abs(finalCardCorners[i].x - baseline[i].x) > MOTION_EPSILON_PX ||
                                                    abs(finalCardCorners[i].y - baseline[i].y) > MOTION_EPSILON_PX
                                        }
                                if (moved) {
                                    stableFrameCount = 1
                                    lastStableCorners = finalCardCorners
                                } else {
                                    stableFrameCount++
                                }

                                if (stableFrameCount >= STABLE_FRAMES_REQUIRED) {
                                    autoCaptureTriggered = true
                                    val sensorFrameCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
                                    val capturedCardCorners = finalCardCorners
                                    val capturedQRCorners = screenQR
                                    val capturedMarkerMap = markerScreenMap
                                    // previewView.bitmap MUST be read on the main thread
                                    coroutineScope.launch(Dispatchers.Main) {
                                        val previewSnapshot = previewView.bitmap
                                        if (previewSnapshot != null) {
                                            processCurrentFrame(
                                                sensorBitmap = sensorFrameCopy,
                                                previewSnapshot = previewSnapshot,
                                                qrValue = qrValue,
                                                qrPoints = qrImagePoints,
                                                cardCornersScreen = capturedCardCorners,
                                                qrCornersScreen = capturedQRCorners,
                                                markerCentersScreen = capturedMarkerMap
                                            )
                                        } else {
                                            sensorFrameCopy.recycle()
                                            scanStatus = "Preview not ready, try again"
                                        }
                                        autoCaptureTriggered = false
                                    }
                                }
                            } else {
                                stableFrameCount = 0
                                lastStableCorners = null
                                autoCaptureTriggered = false
                            }

                            downsampled.recycle()
                            imageProxy.close()
                        }
                        .addOnFailureListener(cameraExecutor) { e ->
                            Log.e(TAG, "QR detection failed", e)
                            // Fallback to edge detection (no overlay)
                            val cardCorners = CardDetector.detectCardCorners(downsampled)
                            if (cardCorners != null && cardCorners.size == 4) {
                                val scaleX = imageWidth / processWidth
                                val scaleY = imageHeight / processHeight
                                val screenCorners = cardCorners.map { Offset(it.x * scaleX, it.y * scaleY) }
                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = screenCorners
                                    isTracking = true
                                    rawQRCorners = null
                                    qrDetected = false
                                    rawMarkerCenters = null
                                    stableFrameCount = 0
                                    lastStableCorners = null
                                    autoCaptureTriggered = false
                                }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) {
                                    isTracking = false
                                    rawCardCorners = null
                                    rawQRCorners = null
                                    rawMarkerCenters = null
                                    qrDetected = false
                                    stableFrameCount = 0
                                    lastStableCorners = null
                                    autoCaptureTriggered = false
                                }
                            }
                            downsampled.recycle()
                            imageProxy.close()
                        }
                }

                val camera = cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                cameraError = null
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera start error", e)
                cameraError = "Failed to start camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }

    LaunchedEffect(hasCameraPermission) {
        if (hasCameraPermission) startCamera() else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // ─── UI ──────────────────────────────────────────────────────────
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            // Permission / error handling
            if (!hasCameraPermission) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Camera permission required", color = Color.White)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                            Text("Grant Permission")
                        }
                    }
                }
                return@Scaffold
            }

            if (cameraError != null) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(cameraError!!, color = Color.White, textAlign = TextAlign.Center)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { cameraError = null; startCamera() }) {
                            Text("Retry")
                        }
                    }
                }
                return@Scaffold
            }

            // Camera preview + overlays
            Box(modifier = Modifier.fillMaxWidth().height(400.dp).background(Color.Black)) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                StaticViewfinder()
                LiveCardOverlay(
                    cardCorners = smoothCorners,
                    qrCorners = smoothQRCorners,
                    markerCenters = smoothMarkerMap,
                    bubblePositions = null,
                    bubbleStatus = null,
                    isTracking = isTracking,
                    qrDetected = qrDetected,
                    scaleFactor = 1.08f
                )
                if (isScanning || isProcessing) {
                    ScanningLineOverlay(isScanning = true, status = scanStatus, frames = 1, totalFrames = 1)
                }
            }

            // Status text
            val statusText = buildString {
                if (isTracking) append("✅ Card detected") else append("❌ Card not detected")
                if (qrDetected) append(" | QR ✓") else append(" | QR ✗")
                if (rawMarkerCenters != null) append(" | Markers ${rawMarkerCenters!!.size}/4")
                else append(" | Markers 0/4")
            }
            Text(
                text = statusText,
                color = if (isTracking && qrDetected) Color.Green else Color.Yellow,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(8.dp)
            )

            // Controls (manual entry + capture button)
            Column(modifier = Modifier.fillMaxWidth().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                    TextButton(onClick = { showManualEntry = !showManualEntry }) {
                        Text(if (showManualEntry) "Hide Manual Entry" else "Manual Entry")
                    }
                }

                if (showManualEntry) {
                    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            OutlinedTextField(
                                value = manualSerial,
                                onValueChange = { manualSerial = it },
                                label = { Text("Enter Serial Code") },
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true
                            )
                            Spacer(Modifier.height(8.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                Button(
                                    onClick = {
                                        if (manualSerial.isNotBlank()) {
                                            scanResult = LocalScanResult(manualSerial, emptyList(), "{}", 1f)
                                            showManualEntry = false
                                        }
                                    },
                                    modifier = Modifier.weight(1f),
                                    enabled = manualSerial.isNotBlank()
                                ) {
                                    Text("Apply")
                                }
                                Button(onClick = { showManualEntry = false }, modifier = Modifier.weight(1f)) {
                                    Text("Cancel")
                                }
                            }
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))

                Button(
                    onClick = {
                        val token = latestToken
                        val sensorFrame = currentFrameBitmap
                        val previewSnapshot = previewView.bitmap
                        when {
                            token == null || sensorFrame == null -> {
                                scanStatus = "No frame or QR code detected yet"
                            }
                            previewSnapshot == null -> {
                                scanStatus = "Preview not ready, try again"
                            }
                            else -> {
                                processCurrentFrame(
                                    sensorBitmap = sensorFrame,
                                    previewSnapshot = previewSnapshot,
                                    qrValue = token,
                                    qrPoints = currentQrPoints,
                                    cardCornersScreen = smoothCorners,
                                    qrCornersScreen = smoothQRCorners,
                                    markerCentersScreen = smoothMarkerMap
                                )
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(0.7f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFEB914)),
                    enabled = latestToken != null && currentFrameBitmap != null
                ) {
                    Text("Capture & Verify", color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }

            // ─── Result Dialog: Side‑by‑Side ──────────────────────────
            scanResult?.let { result ->
                AlertDialog(
                    onDismissRequest = {
                        scanResult = null
                        isScanning = false
                        result.warpedCardBitmap?.recycle()
                        result.originalBitmap?.recycle()
                    },
                    text = {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Processed (warped)
                                Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Processed", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    if (result.warpedCardBitmap != null) {
                                        Image(
                                            bitmap = result.warpedCardBitmap.asImageBitmap(),
                                            contentDescription = "Processed",
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .height(180.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                        )
                                    } else {
                                        Box(modifier = Modifier.fillMaxWidth().height(180.dp).background(Color.Gray), contentAlignment = Alignment.Center) {
                                            Text("No image", color = Color.White, fontSize = 12.sp)
                                        }
                                    }
                                }
                                // Original (with overlay) — exact preview capture
                                Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Original", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    if (result.originalBitmap != null) {
                                        Image(
                                            bitmap = result.originalBitmap.asImageBitmap(),
                                            contentDescription = "Original with overlay",
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .height(180.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                        )
                                    } else {
                                        Box(modifier = Modifier.fillMaxWidth().height(180.dp).background(Color.Gray), contentAlignment = Alignment.Center) {
                                            Text("No image", color = Color.White, fontSize = 12.sp)
                                        }
                                    }
                                }
                            }

                            // Info row
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Token", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(result.token, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                                }
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Filled", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(result.filledIndices.joinToString(limit = 5), fontSize = 14.sp)
                                }
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Confidence", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(String.format("%.0f%%", result.confidence * 100), fontWeight = FontWeight.Bold)
                                }
                            }

                            var showDetails by remember { mutableStateOf(false) }
                            TextButton(onClick = { showDetails = !showDetails }) {
                                Text(if (showDetails) "Hide raw data" else "Show raw data")
                            }
                            if (showDetails) {
                                Text(
                                    result.scanDataJson,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
                                        .padding(8.dp)
                                )
                            }
                        }
                    },
                    confirmButton = {
                        Button(onClick = {
                            scanResult = null
                            isScanning = false
                            result.warpedCardBitmap?.recycle()
                            result.originalBitmap?.recycle()
                        }) {
                            Text("OK")
                        }
                    }
                )
            }
        }
    }
}

private suspend fun <T> Task<T>.await(): T {
    return suspendCancellableCoroutine { continuation ->
        addOnSuccessListener { result: T -> continuation.resume(result) }
        addOnFailureListener { exception: Exception -> continuation.resumeWithException(exception) }
    }
}