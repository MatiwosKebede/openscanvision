package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.PointF
import android.util.Log
import android.util.Rational
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.openscanvision.omr.CardDetector
import org.openscanvision.ui.components.StaticViewfinder
import java.nio.ByteBuffer
import java.util.concurrent.Executors

private const val TAG = "ArUcoScanner"

// How many consecutive frames tracking is allowed to find nothing before
// we pay for a reacquire scan. Keeps steady-state cost low (small ROIs
// only) while still recovering quickly from lost track.
private const val MAX_FRAMES_BEFORE_RESCAN = 8
private const val TRACK_HALF_SIZE_PX = 70
private const val REACQUIRE_DOWNSCALE = 0.5
// Smoothing applied only to what's drawn on screen (not to the positions
// fed back into tracking/search), to cut jitter without adding search lag.
private const val DISPLAY_SMOOTHING_ALPHA = 0.55f

// ─── Helper functions (copied from ui.utils to avoid imports) ──

/**
 * Rotates points from raw buffer space into the "upright" frame that
 * matches what the user visually sees (i.e. what rotationDegrees corrects for).
 *
 * NOTE: for 90/270 the width/height axes swap. Callers must pass the
 * POST-rotation width/height (not the raw buffer width/height) to any
 * function that scales these points against a target view.
 */
private fun mapBitmapPointsToSensor(
    points: List<PointF>,
    sensorWidth: Float,
    sensorHeight: Float,
    rotationDegrees: Int
): List<PointF> {
    if (rotationDegrees % 360 == 0) return points
    return when (rotationDegrees) {
        90 -> points.map { PointF(sensorHeight - it.y, it.x) }
        180 -> points.map { PointF(sensorWidth - it.x, sensorHeight - it.y) }
        270 -> points.map { PointF(it.y, sensorWidth - it.x) }
        else -> points
    }
}

/**
 * Maps points from a rotation-corrected analysis frame (frameWidth x frameHeight)
 * into PreviewView screen coordinates. Replicates PreviewView's default
 * FILL_CENTER scaleType (uniform scale, center-crop). Switch maxOf -> minOf
 * if you've explicitly set FIT_CENTER.
 */
private fun mapImageToScreen(
    points: List<Offset>,
    frameWidth: Float,
    frameHeight: Float,
    previewView: PreviewView
): List<Offset>? {
    val viewWidth = previewView.width.toFloat()
    val viewHeight = previewView.height.toFloat()
    if (viewWidth <= 0 || viewHeight <= 0 || frameWidth <= 0 || frameHeight <= 0) return null

    val scale = maxOf(viewWidth / frameWidth, viewHeight / frameHeight)
    val scaledW = frameWidth * scale
    val scaledH = frameHeight * scale
    val offsetX = (viewWidth - scaledW) / 2f
    val offsetY = (viewHeight - scaledH) / 2f

    return points.map { Offset(it.x * scale + offsetX, it.y * scale + offsetY) }
}

/**
 * Builds a single-channel grayscale OpenCV Mat directly from the Y-plane
 * of a YUV_420_888 ImageProxy. This is the fast path: it skips YUV->RGB
 * conversion, a Bitmap allocation, and the extra color-conversion that
 * CardDetector's Bitmap-based functions would otherwise redo internally.
 * ArUco detection only ever needs luma, so this is the entire input it needs.
 *
 * Caller owns the returned Mat and must release() it.
 */
private fun yPlaneToGrayMat(imageProxy: ImageProxy): Mat {
    val yPlane = imageProxy.planes[0]
    val buffer: ByteBuffer = yPlane.buffer
    val rowStride = yPlane.rowStride
    val pixelStride = yPlane.pixelStride
    val width = imageProxy.width
    val height = imageProxy.height

    val mat = Mat(height, width, CvType.CV_8UC1)

    if (pixelStride == 1 && rowStride == width) {
        // Fully contiguous — single bulk copy.
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        mat.put(0, 0, bytes)
    } else if (pixelStride == 1) {
        // Row-padded (rowStride > width) but pixels within a row are contiguous.
        val rowBytes = ByteArray(width)
        for (row in 0 until height) {
            buffer.position(row * rowStride)
            buffer.get(rowBytes, 0, width)
            mat.put(row, 0, rowBytes)
        }
    } else {
        // Rare: non-unit pixel stride within the Y plane. Fall back to a
        // per-pixel strided copy (still avoids full YUV->RGB conversion).
        val rowBytes = ByteArray(rowStride)
        val outRow = ByteArray(width)
        for (row in 0 until height) {
            buffer.position(row * rowStride)
            val remaining = buffer.remaining().coerceAtMost(rowStride)
            buffer.get(rowBytes, 0, remaining)
            for (col in 0 until width) {
                outRow[col] = rowBytes[col * pixelStride]
            }
            mat.put(row, 0, outRow)
        }
    }
    return mat
}

// ─── Composable ──────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    // ─── Permission ──────────────────────────────────────────────

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasCameraPermission = it }

    // ─── UI state ─────────────────────────────────────────────────

    var detectedMarkers by remember { mutableStateOf<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) }
    var isTracking by remember { mutableStateOf(false) }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewLaidOut by remember { mutableStateOf(false) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Analyzer-thread-only state (NOT Compose state — mutated and read
    // exclusively on cameraExecutor, so no recomposition/allocation cost
    // per frame just for tracking bookkeeping).
    val lastKnownCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }       // this frame's raw centres
    val previousCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }        // prior frame's raw centres (for velocity)
    val smoothedMarkersRef = remember { arrayOf<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) } // EMA'd, screen-space
    val framesSinceFullScanRef = remember { intArrayOf(MAX_FRAMES_BEFORE_RESCAN) } // force scan on first frame

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    DisposableEffect(previewView) {
        val listener = android.view.View.OnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            if (!previewLaidOut && previewView.width > 0 && previewView.height > 0) {
                previewLaidOut = true
            }
        }
        previewView.addOnLayoutChangeListener(listener)
        onDispose { previewView.removeOnLayoutChangeListener(listener) }
    }

    // ─── Camera setup ─────────────────────────────────────────────

    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()

                val preview = Preview.Builder().build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    val mediaImage = imageProxy.image
                    if (mediaImage == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val bufferWidth = imageProxy.width.toFloat()
                    val bufferHeight = imageProxy.height.toFloat()
                    val rotatedWidth = if (rotationDegrees % 180 == 0) bufferWidth else bufferHeight
                    val rotatedHeight = if (rotationDegrees % 180 == 0) bufferHeight else bufferWidth

                    // Fast path: grayscale Mat straight from the Y-plane, no Bitmap.
                    val gray = yPlaneToGrayMat(imageProxy)

                    val arUcoMap: Map<Int, List<PointF>>
                    try {
                        val lastCentres = lastKnownCentresRef[0]
                        val prevCentres = previousCentresRef[0]
                        val needsReacquire = lastCentres.isEmpty() ||
                                framesSinceFullScanRef[0] >= MAX_FRAMES_BEFORE_RESCAN

                        arUcoMap = if (!needsReacquire) {
                            // Predict where each marker will be this frame using simple
                            // constant-velocity extrapolation from the last two observed
                            // positions. This keeps the search ROI centered on fast-moving
                            // markers instead of always lagging one frame behind, so we can
                            // afford a smaller (cheaper) ROI while tracking more reliably.
                            val predictedCentres = lastCentres.mapValues { (id, centre) ->
                                val prev = prevCentres[id]
                                if (prev != null) {
                                    PointF(centre.x + (centre.x - prev.x), centre.y + (centre.y - prev.y))
                                } else {
                                    centre
                                }
                            }
                            val tracked = CardDetector.detectArUcoMarkersTrackedInGray(
                                gray, predictedCentres, TRACK_HALF_SIZE_PX
                            )
                            if (tracked.size < lastCentres.size) {
                                // Lost one or more markers this frame — force a reacquire
                                // next frame instead of drifting silently.
                                framesSinceFullScanRef[0] = MAX_FRAMES_BEFORE_RESCAN
                            } else {
                                framesSinceFullScanRef[0]++
                            }
                            tracked
                        } else {
                            // Coarse-to-fine reacquire: cheap downscaled full-frame
                            // localization + full-res sub-pixel refine per hit. Faster
                            // AND more accurate than a single full-res pass.
                            val reacquired = CardDetector.detectArUcoMarkersReacquire(gray, REACQUIRE_DOWNSCALE)
                            framesSinceFullScanRef[0] = 0
                            reacquired
                        }
                    } finally {
                        gray.release()
                    }

                    // Shift tracking state for next frame's velocity estimate.
                    previousCentresRef[0] = lastKnownCentresRef[0]
                    lastKnownCentresRef[0] = arUcoMap.mapValues { (_, corners) ->
                        PointF(corners.map { it.x }.average().toFloat(), corners.map { it.y }.average().toFloat())
                    }

                    // Map corners to screen coordinates (raw, unsmoothed — this is what
                    // feeds next frame's search, so it must reflect the true detection).
                    val rawScreenMarkers = mutableMapOf<Int, Pair<Offset, List<Offset>>>()
                    for ((id, corners) in arUcoMap) {
                        val sensorCorners = mapBitmapPointsToSensor(corners, bufferWidth, bufferHeight, rotationDegrees)
                        val screenCorners = mapImageToScreen(
                            sensorCorners.map { point -> Offset(point.x, point.y) },
                            rotatedWidth,
                            rotatedHeight,
                            previewView
                        )
                        if (screenCorners != null && screenCorners.size == 4) {
                            val centreX = screenCorners.map { it.x }.average().toFloat()
                            val centreY = screenCorners.map { it.y }.average().toFloat()
                            rawScreenMarkers[id] = Pair(Offset(centreX, centreY), screenCorners)
                        }
                    }

                    // Exponential moving average purely for display — cuts visual jitter
                    // from per-frame corner noise without adding lag to the actual
                    // tracking/search logic above, which always uses raw detections.
                    val prevSmoothed = smoothedMarkersRef[0]
                    val smoothed = rawScreenMarkers.mapValues { (id, raw) ->
                        val prev = prevSmoothed[id]
                        if (prev == null) {
                            raw
                        } else {
                            val (rawCentre, rawCorners) = raw
                            val (prevCentre, prevCorners) = prev
                            val a = DISPLAY_SMOOTHING_ALPHA
                            val newCentre = Offset(
                                prevCentre.x + a * (rawCentre.x - prevCentre.x),
                                prevCentre.y + a * (rawCentre.y - prevCentre.y)
                            )
                            val newCorners = rawCorners.indices.map { i ->
                                Offset(
                                    prevCorners[i].x + a * (rawCorners[i].x - prevCorners[i].x),
                                    prevCorners[i].y + a * (rawCorners[i].y - prevCorners[i].y)
                                )
                            }
                            Pair(newCentre, newCorners)
                        }
                    }
                    smoothedMarkersRef[0] = smoothed

                    coroutineScope.launch(Dispatchers.Main) {
                        detectedMarkers = smoothed
                        isTracking = smoothed.isNotEmpty()
                    }

                    imageProxy.close()
                }

                val existingViewPort = previewView.viewPort
                val viewPort = existingViewPort ?: run {
                    val w = previewView.width
                    val h = previewView.height
                    if (w > 0 && h > 0) {
                        ViewPort.Builder(
                            Rational(w, h),
                            previewView.display?.rotation ?: android.view.Surface.ROTATION_0
                        ).build()
                    } else {
                        null
                    }
                }

                if (viewPort != null) {
                    val useCaseGroup = UseCaseGroup.Builder()
                        .addUseCase(preview)
                        .addUseCase(analysis)
                        .setViewPort(viewPort)
                        .build()

                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup
                    )
                } else {
                    Log.w(TAG, "PreviewView not laid out yet, binding without ViewPort")
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                    )
                }

                cameraError = null
                Log.d(TAG, "Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera start error", e)
                cameraError = "Failed to start camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }

    LaunchedEffect(hasCameraPermission, previewLaidOut) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        } else if (previewLaidOut) {
            startCamera()
        }
    }

    // ─── Live preview overlay ─────────────────────────────────────

    @Composable
    fun ArUcoOverlay(markers: Map<Int, Pair<Offset, List<Offset>>>) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            markers.forEach { (id, data) ->
                val (centre, corners) = data

                val path = Path().apply {
                    moveTo(corners[0].x, corners[0].y)
                    lineTo(corners[1].x, corners[1].y)
                    lineTo(corners[2].x, corners[2].y)
                    lineTo(corners[3].x, corners[3].y)
                    close()
                }
                drawPath(
                    path = path,
                    color = Color.Green,
                    style = Stroke(width = 4f)
                )

                drawCircle(
                    center = centre,
                    radius = 8f,
                    color = Color.Red
                )

                val text = "ID: $id  X: ${centre.x.toInt()}  Y: ${centre.y.toInt()}"
                drawContext.canvas.nativeCanvas.apply {
                    val paint = android.graphics.Paint().apply {
                        color = android.graphics.Color.WHITE
                        textSize = 40f
                        isAntiAlias = true
                        setShadowLayer(4f, 2f, 2f, android.graphics.Color.BLACK)
                    }
                    drawText(text, centre.x + 15f, centre.y - 15f, paint)
                }
            }
        }
    }

    // ─── UI Scaffold ──────────────────────────────────────────────

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = { Text("ArUco Scanner") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (!hasCameraPermission) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Camera permission required")
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
                        Text(cameraError!!, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { cameraError = null; startCamera() }) { Text("Retry") }
                    }
                }
                return@Scaffold
            }

            Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                StaticViewfinder()
                ArUcoOverlay(markers = detectedMarkers)

                Text(
                    text = if (isTracking) "✅ ${detectedMarkers.size} marker(s) detected" else "❌ No markers",
                    color = if (isTracking) Color.Green else Color.Yellow,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(16.dp)
                        .background(Color.Black.copy(alpha = 0.6f), shape = MaterialTheme.shapes.small)
                        .padding(8.dp)
                )
            }
        }
    }
}