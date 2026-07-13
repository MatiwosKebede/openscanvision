package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Color as AndroidColor
import android.graphics.Paint as AndroidPaint
import android.graphics.Path as AndroidPath
import android.graphics.PointF
import android.util.Log
import android.util.Rational
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.openscanvision.omr.CardDetector
import org.openscanvision.omr.ImagePreprocessor
import org.openscanvision.omr.Templates
import org.openscanvision.omr.OpenCVUtils
import org.openscanvision.omr.toBitmap
import org.openscanvision.ui.components.StaticViewfinder
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.Executors

private const val TAG = "ArUcoScanner"

// ─── Optimised constants ──────────────────────────────────────────

private const val MAX_FRAMES_BEFORE_RESCAN = 120
private const val LOW_CONF_FRAMES_BEFORE_RESCAN = 10
private const val BASE_TRACK_HALF_SIZE = 55          // reduced from 70
private const val MIN_TRACK_HALF_SIZE = 30           // reduced from 40
private const val MAX_TRACK_HALF_SIZE = 100          // reduced from 130
private const val DISPLAY_SMOOTHING_ALPHA = 0.25f
private const val CONFIDENCE_THRESHOLD = 5
private const val MIN_CAPTURE_HIGH_CONF_MARKERS = 3
private const val MIN_CAPTURE_AVG_CONF = 6f
private const val MAX_CAPTURE_HOMOGRAPHY_ERROR_PX = 9f
private const val MAX_MARKER_AREA_RATIO = 3.0f
private const val CONFIDENCE_DECAY = 0.92f
private const val PERSISTENCE_FRAMES = 20
private const val ACCELERATION_NOISE = 0.02f
private const val AUTO_CAPTURE_STABLE_FRAMES = 12
private const val AUTO_CAPTURE_COOLDOWN_FRAMES = 20
private const val QUICK_RETRY_DELAY_FRAMES = 8

// ─── Constant‑acceleration Kalman filter ────────────────────────

private data class KalmanState(
    var x: Float, var y: Float,
    var vx: Float, var vy: Float,
    var ax: Float, var ay: Float,
    var px: Float = 1f, var py: Float = 1f,
    var pvx: Float = 1f, var pvy: Float = 1f,
    var pax: Float = 0.5f, var pay: Float = 0.5f
) {
    fun predict(dt: Float = 1f) {
        x += vx * dt + 0.5f * ax * dt * dt
        y += vy * dt + 0.5f * ay * dt * dt
        vx += ax * dt
        vy += ay * dt
        px += pvx * dt * dt + 0.25f * pax * dt * dt * dt * dt
        py += pvy * dt * dt + 0.25f * pay * dt * dt * dt * dt
        pvx += pax * dt * dt + ACCELERATION_NOISE
        pvy += pay * dt * dt + ACCELERATION_NOISE
        pax += ACCELERATION_NOISE
        pay += ACCELERATION_NOISE
    }

    fun update(measuredX: Float, measuredY: Float) {
        val kx = px / (px + 1f)
        val ky = py / (py + 1f)
        val residualX = measuredX - x
        val residualY = measuredY - y
        x += kx * residualX
        y += ky * residualY
        vx += kx * residualX / 1f
        vy += ky * residualY / 1f
        ax += kx * residualX / 2f
        ay += ky * residualY / 2f
        px = (1f - kx) * px
        py = (1f - ky) * py
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

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

private fun yPlaneToGrayMat(imageProxy: ImageProxy): Mat {
    val yPlane = imageProxy.planes[0]
    val buffer: ByteBuffer = yPlane.buffer
    val rowStride = yPlane.rowStride
    val pixelStride = yPlane.pixelStride
    val width = imageProxy.width
    val height = imageProxy.height

    val mat = Mat(height, width, CvType.CV_8UC1)

    if (pixelStride == 1 && rowStride == width) {
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        mat.put(0, 0, bytes)
    } else if (pixelStride == 1) {
        val rowBytes = ByteArray(width)
        for (row in 0 until height) {
            buffer.position(row * rowStride)
            buffer.get(rowBytes, 0, width)
            mat.put(row, 0, rowBytes)
        }
    } else {
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

private fun cardCornersFromArUco(arUcoMap: Map<Int, List<PointF>>): List<PointF>? {
    val homography = CardDetector.buildHomographyFromArUco(arUcoMap) ?: return null
    val templateCardCorners = listOf(
        PointF(0f, 0f),
        PointF((Templates.REF_WIDTH - 1).toFloat(), 0f),
        PointF((Templates.REF_WIDTH - 1).toFloat(), (Templates.REF_HEIGHT - 1).toFloat()),
        PointF(0f, (Templates.REF_HEIGHT - 1).toFloat())
    )
    return CardDetector.predictImagePoints(templateCardCorners, homography)
}

private data class CaptureMetrics(
    var attempts: Int = 0,
    var successes: Int = 0,
    var qualityRejects: Int = 0,
    var totalLockFrames: Long = 0L,
    var totalTimeToCaptureMs: Long = 0L
)

private data class CaptureValidation(
    val accepted: Boolean,
    val reason: String,
    val homographyErrorPx: Float
)

private fun markerArea(corners: List<PointF>): Float {
    if (corners.size != 4) return 0f
    var area = 0f
    for (i in corners.indices) {
        val next = corners[(i + 1) % corners.size]
        area += corners[i].x * next.y - next.x * corners[i].y
    }
    return kotlin.math.abs(area) * 0.5f
}

private fun validateCaptureQuality(
    arUcoMap: Map<Int, List<PointF>>,
    markerConfidence: Map<Int, Int>
): CaptureValidation {
    val highConfMarkers = arUcoMap.keys.count { (markerConfidence[it] ?: 0) >= CONFIDENCE_THRESHOLD }
    if (highConfMarkers < MIN_CAPTURE_HIGH_CONF_MARKERS) {
        return CaptureValidation(false, "Quality low: insufficient stable markers.", Float.MAX_VALUE)
    }

    val avgConfidence = if (arUcoMap.isNotEmpty()) {
        arUcoMap.keys.map { markerConfidence[it] ?: 0 }.average().toFloat()
    } else {
        0f
    }
    if (avgConfidence < MIN_CAPTURE_AVG_CONF) {
        return CaptureValidation(false, "Quality low: marker confidence too low.", Float.MAX_VALUE)
    }

    val areas = arUcoMap.values.map(::markerArea).filter { it > 1f }
    if (areas.size >= 2) {
        val minArea = areas.minOrNull() ?: 0f
        val maxArea = areas.maxOrNull() ?: 0f
        if (minArea <= 0f || maxArea / minArea > MAX_MARKER_AREA_RATIO) {
            return CaptureValidation(false, "Quality low: marker consistency check failed.", Float.MAX_VALUE)
        }
    }

    val homographyError = CardDetector.computeArUcoHomographyError(arUcoMap)
    if (homographyError == null || homographyError > MAX_CAPTURE_HOMOGRAPHY_ERROR_PX) {
        return CaptureValidation(false, "Quality low: homography reprojection error is high.", homographyError ?: Float.MAX_VALUE)
    }

    return CaptureValidation(true, "", homographyError)
}

private fun drawOverlayCardBitmap(source: Bitmap, cardCorners: List<PointF>): Bitmap {
    val mutable = source.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = AndroidCanvas(mutable)
    val strokePaint = AndroidPaint().apply {
        color = AndroidColor.GREEN
        style = AndroidPaint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }
    val pointPaint = AndroidPaint().apply {
        color = AndroidColor.GREEN
        style = AndroidPaint.Style.FILL
        isAntiAlias = true
    }
    val path = AndroidPath().apply {
        moveTo(cardCorners[0].x, cardCorners[0].y)
        lineTo(cardCorners[1].x, cardCorners[1].y)
        lineTo(cardCorners[2].x, cardCorners[2].y)
        lineTo(cardCorners[3].x, cardCorners[3].y)
        close()
    }
    canvas.drawPath(path, strokePaint)
    cardCorners.forEach { canvas.drawCircle(it.x, it.y, 9f, pointPaint) }
    return mutable
}

// ─── Composable ──────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    // Permission
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasCameraPermission = it }

    // UI state
    var detectedMarkers by remember { mutableStateOf<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) }
    var isTracking by remember { mutableStateOf(false) }
    var overlayCaptureBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var transformedCaptureBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var fullScreenBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var captureStatus by remember { mutableStateOf("Tap Capture Preview to generate side-by-side card images.") }
    var metricsLine by remember { mutableStateOf("Attempts: 0 | Success: 0 | Avg lock: 0f | Avg time: 0ms | Reject: 0.0%") }
    val captureRequestedRef = remember { AtomicBoolean(false) }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewLaidOut by remember { mutableStateOf(false) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Analyzer state
    val kalmanFilters = remember { mutableMapOf<Int, KalmanState>() }
    val markerConfidence = remember { mutableMapOf<Int, Int>() }
    val markerAge = remember { mutableMapOf<Int, Int>() }
    val lastKnownCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }
    val previousCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }
    val smoothedMarkersRef = remember { arrayOf<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) }
    val framesSinceFullScanRef = remember { intArrayOf(MAX_FRAMES_BEFORE_RESCAN) }
    val stableFrameCounter = remember { intArrayOf(0) }
    val autoCaptureCooldownRef = remember { intArrayOf(0) }
    val quickRetryDelayRef = remember { intArrayOf(0) }
    val retryPendingRef = remember { intArrayOf(0) }
    val lockStartFrameRef = remember { intArrayOf(-1) }
    val lockStartTimeMsRef = remember { longArrayOf(0L) }
    val metricsRef = remember { CaptureMetrics() }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
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

                // Lower resolution for speed (480x360)
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(480, 360))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                var frameCounter = 0
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    frameCounter++

                    if (imageProxy.image == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val bufferWidth = imageProxy.width.toFloat()
                    val bufferHeight = imageProxy.height.toFloat()
                    val rotatedWidth = if (rotationDegrees % 180 == 0) bufferWidth else bufferHeight
                    val rotatedHeight = if (rotationDegrees % 180 == 0) bufferHeight else bufferWidth

                    val gray = yPlaneToGrayMat(imageProxy)

                    val lastCentres = lastKnownCentresRef[0]
                    val prevCentres = previousCentresRef[0]

                    var arUcoMap: Map<Int, List<PointF>> = emptyMap()
                    try {
                        val capturePending = captureRequestedRef.get()
                        val highConfidenceTracked = markerConfidence.count { it.value >= CONFIDENCE_THRESHOLD }
                        val lowConfidenceTracking = highConfidenceTracked < MIN_CAPTURE_HIGH_CONF_MARKERS
                        val shouldTryTracking = lastCentres.isNotEmpty()

                        if (shouldTryTracking) {
                            val allActiveIds = lastCentres.keys + markerAge.filter { it.value < PERSISTENCE_FRAMES }.keys
                            val predictedCentres = allActiveIds.associateWith { id ->
                                val kalman = kalmanFilters[id]
                                if (kalman != null) {
                                    kalman.predict()
                                    PointF(kalman.x, kalman.y)
                                } else {
                                    lastCentres[id] ?: prevCentres[id] ?: PointF(0f, 0f)
                                }
                            }

                            // Smaller ROI for high-confidence markers
                            val halfSizeMap = allActiveIds.associateWith { id ->
                                val conf = markerConfidence[id] ?: 0
                                when {
                                    conf >= CONFIDENCE_THRESHOLD -> MIN_TRACK_HALF_SIZE
                                    conf >= 2 -> BASE_TRACK_HALF_SIZE
                                    else -> MAX_TRACK_HALF_SIZE
                                }
                            }

                            val tracked = CardDetector.detectArUcoMarkersTrackedInGray(
                                gray, predictedCentres, halfSizeMap
                            )

                            val templatePoints = Templates.SHARED_MARKER_CENTRES
                            val filtered = if (tracked.size >= 4) {
                                CardDetector.rejectOutliersWithHomography(tracked, templatePoints)
                            } else tracked

                            if (filtered.size >= allActiveIds.size / 2) {
                                framesSinceFullScanRef[0]++
                            } else {
                                framesSinceFullScanRef[0] = (framesSinceFullScanRef[0] + LOW_CONF_FRAMES_BEFORE_RESCAN)
                                    .coerceAtMost(MAX_FRAMES_BEFORE_RESCAN)
                            }
                            arUcoMap = filtered
                        }

                        val shouldReacquire = arUcoMap.size < MIN_CAPTURE_HIGH_CONF_MARKERS &&
                                (
                                    !shouldTryTracking ||
                                            capturePending ||
                                            (lowConfidenceTracking && framesSinceFullScanRef[0] >= LOW_CONF_FRAMES_BEFORE_RESCAN) ||
                                            framesSinceFullScanRef[0] >= MAX_FRAMES_BEFORE_RESCAN
                                    )

                        if (shouldReacquire) {
                            val reacquired = CardDetector.detectArUcoMarkersReacquire(gray, listOf(0.4, 0.7, 1.0))
                            val templatePoints = Templates.SHARED_MARKER_CENTRES
                            arUcoMap = if (reacquired.size >= MIN_CAPTURE_HIGH_CONF_MARKERS) {
                                CardDetector.rejectOutliersWithHomography(reacquired, templatePoints)
                            } else reacquired
                            framesSinceFullScanRef[0] = 0
                        }
                    } finally {
                        gray.release()
                    }

                    // Update state
                    val detectedIds = arUcoMap.keys
                    for (id in markerAge.keys) {
                        markerAge[id] = (markerAge[id] ?: 0) + 1
                    }
                    for (id in detectedIds) {
                        val corners = arUcoMap[id] ?: continue
                        var cx = 0f
                        var cy = 0f
                        for (corner in corners) {
                            cx += corner.x
                            cy += corner.y
                        }
                        val count = corners.size.coerceAtLeast(1).toFloat()
                        cx /= count
                        cy /= count
                        val kalman = kalmanFilters[id]
                        if (kalman != null) {
                            kalman.update(cx, cy)
                        } else {
                            kalmanFilters[id] = KalmanState(cx, cy, 0f, 0f, 0f, 0f)
                        }
                        markerConfidence[id] = (markerConfidence[id] ?: 0) + 1
                        markerAge[id] = 0
                    }
                    for (id in markerConfidence.keys) {
                        if (id !in detectedIds) {
                            markerConfidence[id] = ((markerConfidence[id] ?: 0) * CONFIDENCE_DECAY).toInt().coerceAtLeast(1)
                        }
                    }
                    val toRemove = markerAge.filter { it.value > PERSISTENCE_FRAMES && (markerConfidence[it.key] ?: 0) < 3 }.keys
                    for (id in toRemove) {
                        kalmanFilters.remove(id)
                        markerConfidence.remove(id)
                        markerAge.remove(id)
                    }

                    previousCentresRef[0] = lastKnownCentresRef[0]
                    lastKnownCentresRef[0] = arUcoMap.mapValues { (_, corners) ->
                        var x = 0f
                        var y = 0f
                        for (corner in corners) {
                            x += corner.x
                            y += corner.y
                        }
                        val count = corners.size.coerceAtLeast(1).toFloat()
                        PointF(x / count, y / count)
                    }

                    val highConfDetected = detectedIds.count { (markerConfidence[it] ?: 0) >= CONFIDENCE_THRESHOLD }
                    val isStableNow = highConfDetected >= MIN_CAPTURE_HIGH_CONF_MARKERS &&
                            arUcoMap.size >= MIN_CAPTURE_HIGH_CONF_MARKERS
                    if (isStableNow) {
                        stableFrameCounter[0]++
                        if (stableFrameCounter[0] == 1) {
                            lockStartFrameRef[0] = frameCounter
                            lockStartTimeMsRef[0] = System.currentTimeMillis()
                        }
                    } else {
                        stableFrameCounter[0] = 0
                        lockStartFrameRef[0] = -1
                        lockStartTimeMsRef[0] = 0L
                    }
                    if (autoCaptureCooldownRef[0] > 0) autoCaptureCooldownRef[0]--
                    if (quickRetryDelayRef[0] > 0) quickRetryDelayRef[0]--

                    if (
                        retryPendingRef[0] == 1 &&
                        quickRetryDelayRef[0] == 0 &&
                        autoCaptureCooldownRef[0] == 0 &&
                        !captureRequestedRef.get() &&
                        isStableNow
                    ) {
                        retryPendingRef[0] = 0
                        captureRequestedRef.set(true)
                        autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                        coroutineScope.launch(Dispatchers.Main) {
                            captureStatus = "Retrying capture..."
                        }
                    }

                    if (
                        stableFrameCounter[0] >= AUTO_CAPTURE_STABLE_FRAMES &&
                        highConfDetected >= MIN_CAPTURE_HIGH_CONF_MARKERS &&
                        autoCaptureCooldownRef[0] == 0 &&
                        !captureRequestedRef.get()
                    ) {
                        captureRequestedRef.set(true)
                        autoCaptureCooldownRef[0] = AUTO_CAPTURE_COOLDOWN_FRAMES
                        coroutineScope.launch(Dispatchers.Main) {
                            captureStatus = "Auto-capturing on stable card..."
                        }
                    }

                    // Map to screen
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

                    if (captureRequestedRef.compareAndSet(true, false)) {
                        metricsRef.attempts++
                        val captureValidation = validateCaptureQuality(arUcoMap, markerConfidence)
                        if (!captureValidation.accepted) {
                            metricsRef.qualityRejects++
                            retryPendingRef[0] = 1
                            quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                            autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                            coroutineScope.launch(Dispatchers.Main) {
                                captureStatus = "${captureValidation.reason} Auto-retrying..."
                                val attempts = metricsRef.attempts.coerceAtLeast(1)
                                val rejectRate = (metricsRef.qualityRejects * 100f) / attempts
                                metricsLine =
                                    "Attempts: ${metricsRef.attempts} | Success: ${metricsRef.successes} | Avg lock: ${
                                        if (metricsRef.successes > 0) metricsRef.totalLockFrames.toFloat() / metricsRef.successes else 0f
                                    }f | Avg time: ${
                                        if (metricsRef.successes > 0) metricsRef.totalTimeToCaptureMs / metricsRef.successes else 0L
                                    }ms | Reject: ${"%.1f".format(rejectRate)}%"
                            }
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val frameBitmap = imageProxy.toBitmap()
                        if (frameBitmap == null) {
                            retryPendingRef[0] = 1
                            quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                            autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                            coroutineScope.launch(Dispatchers.Main) {
                                captureStatus = "Capture failed: frame conversion failed. Auto-retrying..."
                            }
                        } else {
                            val cardCorners = cardCornersFromArUco(arUcoMap) ?: CardDetector.detectCardCorners(frameBitmap)
                            if (cardCorners == null || cardCorners.size != 4) {
                                retryPendingRef[0] = 1
                                quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                                autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                                coroutineScope.launch(Dispatchers.Main) {
                                    captureStatus = "Capture failed: unable to estimate full card shape. Auto-retrying..."
                                }
                            } else {
                                val overlayBitmap = drawOverlayCardBitmap(frameBitmap, cardCorners)
                                val warped = OpenCVUtils.warpCard(
                                    frameBitmap,
                                    cardCorners,
                                    Templates.REF_WIDTH,
                                    Templates.REF_HEIGHT
                                )
                                val standardized = warped?.let { warpedBitmap ->
                                    val contrasted = ImagePreprocessor.enhanceContrast(warpedBitmap)
                                    ImagePreprocessor.denoise(contrasted)
                                }

                                coroutineScope.launch(Dispatchers.Main) {
                                    overlayCaptureBitmap = overlayBitmap
                                    transformedCaptureBitmap = standardized ?: warped
                                    captureStatus = if (standardized != null || warped != null) {
                                        val lockFrames = if (lockStartFrameRef[0] >= 0) {
                                            (frameCounter - lockStartFrameRef[0]).coerceAtLeast(0)
                                        } else 0
                                        val lockTimeMs = if (lockStartTimeMsRef[0] > 0L) {
                                            System.currentTimeMillis() - lockStartTimeMsRef[0]
                                        } else 0L
                                        metricsRef.successes++
                                        metricsRef.totalLockFrames += lockFrames.toLong()
                                        metricsRef.totalTimeToCaptureMs += lockTimeMs
                                        retryPendingRef[0] = 0
                                        val attempts = metricsRef.attempts.coerceAtLeast(1)
                                        val rejectRate = (metricsRef.qualityRejects * 100f) / attempts
                                        metricsLine =
                                            "Attempts: ${metricsRef.attempts} | Success: ${metricsRef.successes} | Avg lock: ${
                                                if (metricsRef.successes > 0) metricsRef.totalLockFrames.toFloat() / metricsRef.successes else 0f
                                            }f | Avg time: ${
                                                if (metricsRef.successes > 0) metricsRef.totalTimeToCaptureMs / metricsRef.successes else 0L
                                            }ms | Reject: ${"%.1f".format(rejectRate)}%"
                                        "✅ Captured (${lockTimeMs}ms, lock ${lockFrames}f, hErr ${
                                            "%.1f".format(captureValidation.homographyErrorPx)
                                        }px)."
                                    } else {
                                        retryPendingRef[0] = 1
                                        quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                                        autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                                        "Capture failed: perspective transform could not be generated. Auto-retrying..."
                                    }
                                }
                            }
                        }
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
                    } else null
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

                Column(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = if (isTracking) "✅ ${detectedMarkers.size} marker(s) detected" else "❌ No markers",
                        color = if (isTracking) Color.Green else Color.Yellow,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .background(Color.Black.copy(alpha = 0.6f), shape = MaterialTheme.shapes.small)
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                    Spacer(Modifier.height(8.dp))
                    Button(
                        onClick = {
                            if (isTracking) {
                                captureStatus = "Capturing..."
                                captureRequestedRef.set(true)
                            } else {
                                captureStatus = "Hold the card steady first, then capture."
                            }
                        }
                    ) { Text("Capture Preview") }
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = captureStatus,
                        color = Color.White,
                        fontSize = 13.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color.Black.copy(alpha = 0.55f), shape = MaterialTheme.shapes.small)
                            .padding(8.dp)
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = metricsLine,
                        color = Color.LightGray,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color.Black.copy(alpha = 0.45f), shape = MaterialTheme.shapes.small)
                            .padding(horizontal = 8.dp, vertical = 6.dp)
                    )
                    if (overlayCaptureBitmap != null && transformedCaptureBitmap != null) {
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Image(
                                bitmap = overlayCaptureBitmap!!.asImageBitmap(),
                                contentDescription = "Original with overlay",
                                modifier = Modifier
                                    .size(width = 220.dp, height = 140.dp)
                                    .clickable { fullScreenBitmap = overlayCaptureBitmap }
                                    .background(Color.Black)
                            )
                            Image(
                                bitmap = transformedCaptureBitmap!!.asImageBitmap(),
                                contentDescription = "Perspective transformed",
                                modifier = Modifier
                                    .size(width = 220.dp, height = 140.dp)
                                    .clickable { fullScreenBitmap = transformedCaptureBitmap }
                                    .background(Color.Black)
                            )
                        }
                    }
                }
            }
        }
    }

    if (fullScreenBitmap != null) {
        var zoom by remember(fullScreenBitmap) { mutableFloatStateOf(1f) }
        var pan by remember(fullScreenBitmap) { mutableStateOf(Offset.Zero) }

        Dialog(
            onDismissRequest = { fullScreenBitmap = null },
            properties = DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
            ) {
                Image(
                    bitmap = fullScreenBitmap!!.asImageBitmap(),
                    contentDescription = "Zoomable full screen image",
                    modifier = Modifier
                        .fillMaxSize()
                        .clipToBounds()
                        .pointerInput(fullScreenBitmap) {
                            detectTransformGestures { _, dragAmount, zoomChange, _ ->
                                zoom = (zoom * zoomChange).coerceIn(1f, 5f)
                                pan = if (zoom > 1f) pan + dragAmount else Offset.Zero
                            }
                        }
                        .graphicsLayer(
                            scaleX = zoom,
                            scaleY = zoom,
                            translationX = pan.x,
                            translationY = pan.y
                        )
                )
                TextButton(
                    onClick = { fullScreenBitmap = null },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(12.dp)
                ) { Text("Close", color = Color.White) }
            }
        }
    }
}