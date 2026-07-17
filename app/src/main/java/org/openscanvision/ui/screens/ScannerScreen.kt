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
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.asImageBitmap
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
import org.openscanvision.core.internal.omr.CardDetector
import org.openscanvision.core.internal.omr.ImagePreprocessor
import org.openscanvision.core.internal.omr.Templates
import org.openscanvision.core.internal.omr.OMRExtractor
import org.openscanvision.core.internal.omr.OpenCVUtils
import org.openscanvision.core.internal.qr.QrDecoder
import org.openscanvision.ui.theme.NIBGold
import org.openscanvision.utils.toBitmap
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.Executors

private const val TAG = "ScannerScreen"
private const val MAX_FRAMES_BEFORE_RESCAN = 120
private const val LOW_CONF_FRAMES_BEFORE_RESCAN = 10
private const val BASE_TRACK_HALF_SIZE = 55
private const val MIN_TRACK_HALF_SIZE = 30
private const val MAX_TRACK_HALF_SIZE = 100
private const val DISPLAY_SMOOTHING_ALPHA = 0.25f
private const val CONFIDENCE_THRESHOLD = 5
private const val CAPTURE_CONFIDENCE_THRESHOLD = 1
private const val MIN_CAPTURE_HIGH_CONF_MARKERS = 3
private const val REQUIRED_MARKERS_FOR_CAPTURE = 4
private const val PREDICTED_MARKER_SEARCH_HALF_SIZE = 90
private const val MIN_CAPTURE_AVG_CONF = 1f
private const val MAX_CAPTURE_HOMOGRAPHY_ERROR_PX = 12f
private const val MAX_MARKER_AREA_RATIO = 3.0f
private const val CONFIDENCE_DECAY = 0.92f
private const val PERSISTENCE_FRAMES = 20
private const val ACCELERATION_NOISE = 0.02f
private const val AUTO_CAPTURE_STABLE_FRAMES = 1
private const val AUTO_CAPTURE_COOLDOWN_FRAMES = 4
private const val QUICK_RETRY_DELAY_FRAMES = 2

// ─── Kalman filter ──────────────────────────────────────────────────
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

// ─── Helpers ────────────────────────────────────────────────────────
private fun mapBitmapPointsToSensor(
    points: List<PointF>, sensorWidth: Float, sensorHeight: Float, rotationDegrees: Int
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
    points: List<Offset>, frameWidth: Float, frameHeight: Float, previewView: PreviewView
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
            for (col in 0 until width) outRow[col] = rowBytes[col * pixelStride]
            mat.put(row, 0, outRow)
        }
    }
    return mat
}

private fun centreOfCorners(corners: List<PointF>): PointF {
    var x = 0f; var y = 0f
    for (c in corners) { x += c.x; y += c.y }
    val n = corners.size.coerceAtLeast(1).toFloat()
    return PointF(x / n, y / n)
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
    var attempts: Int = 0, var successes: Int = 0, var qualityRejects: Int = 0,
    var totalLockFrames: Long = 0L, var totalTimeToCaptureMs: Long = 0L
)

private data class CaptureValidation(val accepted: Boolean, val reason: String, val homographyErrorPx: Float)

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
    arUcoMap: Map<Int, List<PointF>>, markerConfidence: Map<Int, Int>
): CaptureValidation {
    val highConfMarkers = arUcoMap.keys.count { (markerConfidence[it] ?: 0) >= CAPTURE_CONFIDENCE_THRESHOLD }
    if (highConfMarkers < REQUIRED_MARKERS_FOR_CAPTURE)
        return CaptureValidation(false, "Need all 4 markers stable (have $highConfMarkers).", Float.MAX_VALUE)
    val avgConfidence = arUcoMap.keys.map { markerConfidence[it] ?: 0 }.average().toFloat()
    if (avgConfidence < MIN_CAPTURE_AVG_CONF)
        return CaptureValidation(false, "Marker confidence too low.", Float.MAX_VALUE)
    val areas = arUcoMap.values.map(::markerArea).filter { it > 1f }
    if (areas.size >= 2) {
        val minArea = areas.minOrNull() ?: 0f
        val maxArea = areas.maxOrNull() ?: 0f
        if (minArea <= 0f || maxArea / minArea > MAX_MARKER_AREA_RATIO)
            return CaptureValidation(false, "Marker consistency check failed.", Float.MAX_VALUE)
    }
    val homographyError = CardDetector.computeArUcoHomographyError(arUcoMap)
    if (homographyError == null || homographyError > MAX_CAPTURE_HOMOGRAPHY_ERROR_PX)
        return CaptureValidation(false, "Homography error too high.", homographyError ?: Float.MAX_VALUE)
    return CaptureValidation(true, "", homographyError)
}

private fun drawOverlayCardBitmap(source: Bitmap, cardCorners: List<PointF>): Bitmap {
    val mutable = source.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = AndroidCanvas(mutable)
    val strokePaint = AndroidPaint().apply {
        color = AndroidColor.GREEN; style = AndroidPaint.Style.STROKE; strokeWidth = 8f; isAntiAlias = true
    }
    val pointPaint = AndroidPaint().apply {
        color = AndroidColor.GREEN; style = AndroidPaint.Style.FILL; isAntiAlias = true
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

// ─── Candidate name mapping ─────────────────────────────────────
private val candidateNames = mapOf(
    1 to "Dr. Sintayehu W.",
    2 to "Amanuel Gebre",
    3 to "Tigist Hailu",
    4 to "Michael Chen",
    5 to "Liil Taye",
    6 to "Abdurahman J.",
    7 to "Selamawit Hailu",
    8 to "Yonas Tekle",
    9 to "Mulugeta W.",
    10 to "Hirut Alemu",
    11 to "Bekele Girma",
    12 to "Eyerusalem K."
)

// ─── Composable ──────────────────────────────────────────────────
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasCameraPermission = it
    }

    var transformedCaptureBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var filledBubbleIndices by remember { mutableStateOf<List<Int>>(emptyList()) }
    var bubbleReadConfidence by remember { mutableStateOf(0f) }
    var bubbleTemplateName by remember { mutableStateOf<String?>(null) }
    var qrTokenText by remember { mutableStateOf<String?>(null) }
    var captureStatus by remember { mutableStateOf("Position the card inside the viewfinder to scan.") }
    var metricsLine by remember { mutableStateOf("Attempts: 0 | Success: 0 | Avg lock: 0f | Avg time: 0ms | Reject: 0.0%") }
    val captureRequestedRef = remember { AtomicBoolean(false) }

    var voterId by remember { mutableStateOf("") }
    var voterName by remember { mutableStateOf("") }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewLaidOut by remember { mutableStateOf(false) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val kalmanFilters = remember { mutableMapOf<Int, KalmanState>() }
    val markerConfidence = remember { mutableMapOf<Int, Int>() }
    val markerAge = remember { mutableMapOf<Int, Int>() }
    val lastKnownCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }
    val previousCentresRef = remember { arrayOf<Map<Int, PointF>>(emptyMap()) }
    val framesSinceFullScanRef = remember { intArrayOf(MAX_FRAMES_BEFORE_RESCAN) }
    val stableFrameCounter = remember { intArrayOf(0) }
    val autoCaptureCooldownRef = remember { intArrayOf(0) }
    val quickRetryDelayRef = remember { intArrayOf(0) }
    val retryPendingRef = remember { intArrayOf(0) }
    val lockStartFrameRef = remember { intArrayOf(-1) }
    val lockStartTimeMsRef = remember { longArrayOf(0L) }
    val metricsRef = remember { CaptureMetrics() }

    DisposableEffect(Unit) { onDispose { cameraExecutor.shutdown() } }

    DisposableEffect(previewView) {
        val listener = android.view.View.OnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            if (!previewLaidOut && previewView.width > 0 && previewView.height > 0) previewLaidOut = true
        }
        previewView.addOnLayoutChangeListener(listener)
        onDispose { previewView.removeOnLayoutChangeListener(listener) }
    }

    LaunchedEffect(hasCameraPermission, previewLaidOut) {
        if (hasCameraPermission && previewLaidOut) {
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()
            try {
                cameraProvider.unbindAll()
                val preview = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(640, 360))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                var frameCounter = 0
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    frameCounter++
                    if (imageProxy.image == null) { imageProxy.close(); return@setAnalyzer }
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
                                if (kalman != null) { kalman.predict(); PointF(kalman.x, kalman.y) }
                                else lastCentres[id] ?: prevCentres[id] ?: PointF(0f, 0f)
                            }
                            val halfSizeMap = allActiveIds.associateWith { id ->
                                val conf = markerConfidence[id] ?: 0
                                when {
                                    conf >= CONFIDENCE_THRESHOLD -> MIN_TRACK_HALF_SIZE
                                    conf >= 2 -> BASE_TRACK_HALF_SIZE
                                    else -> MAX_TRACK_HALF_SIZE
                                }
                            }
                            val tracked = CardDetector.detectArUcoMarkersTrackedInGray(gray, predictedCentres, halfSizeMap)
                            val templatePoints = Templates.SHARED_MARKER_CENTRES
                            val filtered = if (tracked.size >= 4) CardDetector.rejectOutliersWithHomography(tracked, templatePoints) else tracked
                            if (filtered.size >= allActiveIds.size / 2) framesSinceFullScanRef[0]++ else framesSinceFullScanRef[0] = (framesSinceFullScanRef[0] + LOW_CONF_FRAMES_BEFORE_RESCAN).coerceAtMost(MAX_FRAMES_BEFORE_RESCAN)
                            arUcoMap = filtered
                        }
                        val shouldReacquire = arUcoMap.size < MIN_CAPTURE_HIGH_CONF_MARKERS && (!shouldTryTracking || capturePending || (lowConfidenceTracking && framesSinceFullScanRef[0] >= LOW_CONF_FRAMES_BEFORE_RESCAN) || framesSinceFullScanRef[0] >= MAX_FRAMES_BEFORE_RESCAN)
                        if (shouldReacquire) {
                            val reacquired = CardDetector.detectArUcoMarkersReacquire(gray, listOf(0.4, 1.0), earlyBreakCount = 3)
                            val templatePoints = Templates.SHARED_MARKER_CENTRES
                            arUcoMap = if (reacquired.size >= MIN_CAPTURE_HIGH_CONF_MARKERS) CardDetector.rejectOutliersWithHomography(reacquired, templatePoints) else reacquired
                            framesSinceFullScanRef[0] = 0
                        }
                        if (arUcoMap.size in 2 until REQUIRED_MARKERS_FOR_CAPTURE) {
                            val knownCentres = arUcoMap.mapValues { (_, c) -> centreOfCorners(c) }
                            val predictions = CardDetector.predictMissingMarkerCentres(knownCentres)
                            if (predictions.isNotEmpty()) {
                                val halfSizeMap = predictions.keys.associateWith { PREDICTED_MARKER_SEARCH_HALF_SIZE }
                                val recovered = CardDetector.detectArUcoMarkersTrackedInGray(gray, predictions, halfSizeMap)
                                if (recovered.isNotEmpty()) arUcoMap = arUcoMap + recovered
                            }
                        }
                    } finally { gray.release() }

                    val detectedIds = arUcoMap.keys
                    for (id in markerAge.keys) markerAge[id] = (markerAge[id] ?: 0) + 1
                    for (id in detectedIds) {
                        val corners = arUcoMap[id] ?: continue
                        var cx = 0f; var cy = 0f
                        for (corner in corners) { cx += corner.x; cy += corner.y }
                        val count = corners.size.coerceAtLeast(1).toFloat()
                        cx /= count; cy /= count
                        val kalman = kalmanFilters[id]
                        if (kalman != null) kalman.update(cx, cy) else kalmanFilters[id] = KalmanState(cx, cy, 0f, 0f, 0f, 0f)
                        markerConfidence[id] = (markerConfidence[id] ?: 0) + 1
                        markerAge[id] = 0
                    }
                    for (id in markerConfidence.keys) {
                        if (id !in detectedIds) markerConfidence[id] = ((markerConfidence[id] ?: 0) * CONFIDENCE_DECAY).toInt().coerceAtLeast(1)
                    }
                    val toRemove = markerAge.filter { it.value > PERSISTENCE_FRAMES && (markerConfidence[it.key] ?: 0) < 3 }.keys
                    for (id in toRemove) {
                        kalmanFilters.remove(id); markerConfidence.remove(id); markerAge.remove(id)
                    }
                    previousCentresRef[0] = lastKnownCentresRef[0]
                    lastKnownCentresRef[0] = arUcoMap.mapValues { (_, corners) ->
                        var x = 0f; var y = 0f
                        for (corner in corners) { x += corner.x; y += corner.y }
                        val count = corners.size.coerceAtLeast(1).toFloat()
                        PointF(x / count, y / count)
                    }
                    val highConfDetected = detectedIds.count { (markerConfidence[it] ?: 0) >= CAPTURE_CONFIDENCE_THRESHOLD }
                    val isStableNow = highConfDetected >= REQUIRED_MARKERS_FOR_CAPTURE && arUcoMap.size >= REQUIRED_MARKERS_FOR_CAPTURE
                    if (isStableNow) {
                        stableFrameCounter[0]++
                        if (stableFrameCounter[0] == 1) { lockStartFrameRef[0] = frameCounter; lockStartTimeMsRef[0] = System.currentTimeMillis() }
                    } else { stableFrameCounter[0] = 0; lockStartFrameRef[0] = -1; lockStartTimeMsRef[0] = 0L }
                    if (autoCaptureCooldownRef[0] > 0) autoCaptureCooldownRef[0]--
                    if (quickRetryDelayRef[0] > 0) quickRetryDelayRef[0]--
                    if (retryPendingRef[0] == 1 && quickRetryDelayRef[0] == 0 && autoCaptureCooldownRef[0] == 0 && !captureRequestedRef.get() && isStableNow) {
                        retryPendingRef[0] = 0; captureRequestedRef.set(true); autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                        coroutineScope.launch(Dispatchers.Main) { captureStatus = "Retrying capture..." }
                    }
                    if (stableFrameCounter[0] >= AUTO_CAPTURE_STABLE_FRAMES && highConfDetected >= REQUIRED_MARKERS_FOR_CAPTURE && autoCaptureCooldownRef[0] == 0 && !captureRequestedRef.get()) {
                        captureRequestedRef.set(true); autoCaptureCooldownRef[0] = AUTO_CAPTURE_COOLDOWN_FRAMES
                        coroutineScope.launch(Dispatchers.Main) { captureStatus = "Auto-capturing on stable card..." }
                    }

                    // ─── CAPTURE BLOCK ──────────────────────────────────────
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
                                metricsLine = "Attempts: ${metricsRef.attempts} | Success: ${metricsRef.successes} | Avg lock: ${if (metricsRef.successes > 0) metricsRef.totalLockFrames.toFloat() / metricsRef.successes else 0f}f | Avg time: ${if (metricsRef.successes > 0) metricsRef.totalTimeToCaptureMs / metricsRef.successes else 0L}ms | Reject: ${"%.1f".format(rejectRate)}%"
                            }
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val frameBitmap = imageProxy.toBitmap()
                        if (frameBitmap == null) {
                            retryPendingRef[0] = 1
                            quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                            autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                            coroutineScope.launch(Dispatchers.Main) { captureStatus = "Frame conversion failed. Auto-retrying..." }
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val cardCorners = cardCornersFromArUco(arUcoMap) ?: CardDetector.detectCardCorners(frameBitmap)
                        if (cardCorners == null || cardCorners.size != 4) {
                            retryPendingRef[0] = 1
                            quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                            autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                            coroutineScope.launch(Dispatchers.Main) { captureStatus = "Card shape estimation failed. Auto-retrying..." }
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val warped = OpenCVUtils.warpCard(frameBitmap, cardCorners, Templates.REF_WIDTH, Templates.REF_HEIGHT)
                        if (warped == null) {
                            retryPendingRef[0] = 1
                            quickRetryDelayRef[0] = QUICK_RETRY_DELAY_FRAMES
                            autoCaptureCooldownRef[0] = QUICK_RETRY_DELAY_FRAMES
                            coroutineScope.launch(Dispatchers.Main) { captureStatus = "Warping failed. Auto-retrying..." }
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val cleaned = ImagePreprocessor.enhanceContrast(warped)
                        val standardized = ImagePreprocessor.denoise(cleaned)

                        coroutineScope.launch {
                            val template = Templates.CANDIDATE

                            val omrDeferred = async(Dispatchers.Default) {
                                OMRExtractor.readBubbleGroupsDetailed(standardized, template)
                            }
                            val qrDeferred = async(Dispatchers.Default) {
                                QrDecoder.decodeFromOriginalFrame(
                                    frameBitmap = frameBitmap,
                                    cardCorners = cardCorners,
                                    standardizedFallback = standardized
                                )
                            }

                            val report = omrDeferred.await()
                            val qrResult = qrDeferred.await()
                            val decodedQrText = qrResult?.first

                            val annotatedBitmap = OMRExtractor.renderAnnotatedImage(standardized, template, report, decodedQrText)
                            val filledOneBased = report.allFilled.map { it + 1 }

                            metricsRef.successes++
                            if (lockStartFrameRef[0] != -1) {
                                metricsRef.totalLockFrames += (frameCounter - lockStartFrameRef[0])
                                metricsRef.totalTimeToCaptureMs += (System.currentTimeMillis() - lockStartTimeMsRef[0])
                            }

                            withContext(Dispatchers.Main) {
                                transformedCaptureBitmap = annotatedBitmap
                                filledBubbleIndices = filledOneBased
                                bubbleReadConfidence = report.overallConfidence
                                bubbleTemplateName = template.name
                                qrTokenText = decodedQrText
                                voterId = "VOTER001"
                                voterName = "Abebech Demissie"
                                captureStatus = if (decodedQrText != null) {
                                    "Success! ${report.allFilled.size} marked, QR read."
                                } else {
                                    "Success! ${report.allFilled.size} marked. QR not read."
                                }
                                val attempts = metricsRef.attempts.coerceAtLeast(1)
                                val rejectRate = (metricsRef.qualityRejects * 100f) / attempts
                                metricsLine = "Attempts: ${metricsRef.attempts} | Success: ${metricsRef.successes} | " +
                                        "Avg lock: ${"%.1f".format(metricsRef.totalLockFrames.toFloat() / metricsRef.successes)}f | " +
                                        "Avg time: ${metricsRef.totalTimeToCaptureMs / metricsRef.successes}ms | " +
                                        "Reject: ${"%.1f".format(rejectRate)}%"
                            }
                        }

                        imageProxy.close()
                        return@setAnalyzer
                    }
                    imageProxy.close()
                }
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
                cameraError = "Failed to start camera: ${e.localizedMessage}"
            }
        }
    }

    // ─── UI ──────────────────────────────────────────────────────
    val brandGold = Color(0xFFC9A237)
    val darkBrown = Color(0xFF1A0D02)
    val white = Color.White

    Box(modifier = Modifier.fillMaxSize().background(white)) {
        if (!hasCameraPermission) {
            PermissionScreen(
                onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                brandGold = brandGold,
                background = white,
                cardColor = white,
                textColor = darkBrown
            )
        } else if (cameraError != null) {
            ErrorScreen(
                message = cameraError!!,
                onRetry = { cameraError = null },
                brandGold = brandGold,
                background = white,
                cardColor = white,
                textColor = darkBrown
            )
        } else {
            Box(modifier = Modifier.fillMaxSize()) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                BottomHud(
                    isTracking = true,
                    captureStatus = captureStatus,
                    metricsLine = metricsLine,
                    onForceScan = { captureRequestedRef.set(true) },
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth(),
                    brandGold = brandGold,
                    textColor = darkBrown
                )
            }
        }
    }

    if (transformedCaptureBitmap != null) {
        VerificationDialog(
            omrBitmap = transformedCaptureBitmap!!,
            templateName = bubbleTemplateName,
            confidence = bubbleReadConfidence,
            filledIndices = filledBubbleIndices,
            voterId = voterId,
            voterName = voterName,
            qrToken = qrTokenText,
            onDismiss = { transformedCaptureBitmap = null; qrTokenText = null },
            onAccept = { transformedCaptureBitmap = null; qrTokenText = null },
            brandGold = brandGold,
            cardColor = white,
            background = white,
            textColor = darkBrown
        )
    }
}

// ─── Sub-composables ───────────────────────────────────────────────

@Composable
private fun PermissionScreen(
    onRequestPermission: () -> Unit,
    brandGold: Color,
    background: Color,
    cardColor: Color,
    textColor: Color
) {
    Box(
        modifier = Modifier.fillMaxSize().background(background),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.padding(32.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("📷", fontSize = 48.sp)
                Spacer(Modifier.height(16.dp))
                Text("Camera Permission Required", fontWeight = FontWeight.Bold, fontSize = 20.sp, color = textColor)
                Spacer(Modifier.height(8.dp))
                Text("This app needs camera access to scan voting cards.", color = textColor.copy(alpha = 0.7f), textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Button(
                    onClick = onRequestPermission,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = brandGold)
                ) {
                    Text("Grant Permission", color = Color.White)
                }
            }
        }
    }
}

@Composable
private fun ErrorScreen(
    message: String,
    onRetry: () -> Unit,
    brandGold: Color,
    background: Color,
    cardColor: Color,
    textColor: Color
) {
    Box(
        modifier = Modifier.fillMaxSize().background(background),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.padding(32.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("⚠️", fontSize = 48.sp)
                Spacer(Modifier.height(16.dp))
                Text("Camera Error", fontWeight = FontWeight.Bold, fontSize = 20.sp, color = textColor)
                Spacer(Modifier.height(8.dp))
                Text(message, color = textColor.copy(alpha = 0.7f), textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Button(
                    onClick = onRetry,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = brandGold)
                ) {
                    Text("Retry", color = Color.White)
                }
            }
        }
    }
}

@Composable
private fun BottomHud(
    isTracking: Boolean,
    captureStatus: String,
    metricsLine: String,
    onForceScan: () -> Unit,
    modifier: Modifier = Modifier,
    brandGold: Color,
    textColor: Color
) {
    Column(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(Color.Transparent, Color.White.copy(alpha = 0.95f))
                )
            )
            .padding(horizontal = 24.dp, vertical = 20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Surface(
            shape = RoundedCornerShape(24.dp),
            color = if (isTracking) brandGold else Color(0xFFF57F17),
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            Text(
                text = if (isTracking) "● Card Detected" else "○ Looking for card...",
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                color = Color.White,
                fontWeight = FontWeight.SemiBold,
                fontSize = 14.sp
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            text = captureStatus,
            color = textColor,
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            maxLines = 2,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text = metricsLine,
            color = textColor.copy(alpha = 0.6f),
            fontSize = 11.sp,
            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(12.dp))
        Button(
            onClick = onForceScan,
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = brandGold, contentColor = Color.White),
            modifier = Modifier.fillMaxWidth(0.7f)
        ) {
            Text("Force Scan", fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun VerificationDialog(
    omrBitmap: Bitmap,
    templateName: String?,
    confidence: Float,
    filledIndices: List<Int>,
    voterId: String,
    voterName: String,
    qrToken: String?,
    onDismiss: () -> Unit,
    onAccept: () -> Unit,
    brandGold: Color,
    cardColor: Color,
    background: Color,
    textColor: Color
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(28.dp),
            color = cardColor,
        ) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Scan Result", fontWeight = FontWeight.Bold, fontSize = 22.sp, color = textColor)
                Spacer(Modifier.height(4.dp))
                Text("Template: ${templateName ?: "Unknown"}  •  Confidence: ${"%.0f".format(confidence * 100)}%",
                    color = textColor.copy(alpha = 0.7f), fontSize = 14.sp)
                Spacer(Modifier.height(16.dp))

                if (voterId.isNotEmpty() || voterName.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = background)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            if (voterId.isNotEmpty()) {
                                Text("Voter ID", color = brandGold, fontSize = 12.sp)
                                Text(voterId, fontWeight = FontWeight.Bold, fontSize = 18.sp, color = textColor)
                            }
                            if (voterName.isNotEmpty()) {
                                Spacer(Modifier.height(8.dp))
                                Text("Name", color = brandGold, fontSize = 12.sp)
                                Text(voterName, fontWeight = FontWeight.Medium, fontSize = 18.sp, color = textColor)
                            }
                        }
                    }
                    Spacer(Modifier.height(16.dp))
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (qrToken != null) background else Color(0xFFFFF3E0)
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "QR Token",
                            color = if (qrToken != null) brandGold else Color(0xFFE65100),
                            fontSize = 12.sp
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = qrToken ?: "Not detected",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 16.sp,
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                            color = if (qrToken != null) textColor else Color(0xFFE65100)
                        )
                    }
                }
                Spacer(Modifier.height(16.dp))

                Card(
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    Image(
                        bitmap = omrBitmap.asImageBitmap(),
                        contentDescription = "OMR detection",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .clip(RoundedCornerShape(16.dp))
                    )
                }
                Spacer(Modifier.height(16.dp))

                if (filledIndices.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = background)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text("Marked Candidates", color = brandGold, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                            Spacer(Modifier.height(8.dp))
                            filledIndices.forEach { index ->
                                val name = candidateNames[index] ?: "Unknown"
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = String.format("%02d", index),
                                        color = brandGold,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 18.sp
                                    )
                                    Text(
                                        text = name,
                                        color = textColor,
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                                Spacer(Modifier.height(4.dp))
                            }
                        }
                    }
                    Spacer(Modifier.height(24.dp))
                } else {
                    Text("No bubbles marked", color = textColor.copy(alpha = 0.5f), fontSize = 16.sp)
                    Spacer(Modifier.height(24.dp))
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = textColor)
                    ) {
                        Text("Retake")
                    }
                    Button(
                        onClick = onAccept,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = brandGold, contentColor = Color.White)
                    ) {
                        Text("Accept & Save")
                    }
                }
            }
        }
    }
}