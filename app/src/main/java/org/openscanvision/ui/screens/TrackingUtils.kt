package org.openscanvision.ui.screens

import android.graphics.PointF
import android.util.Log
import androidx.camera.core.ImageProxy
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.openscanvision.core.internal.omr.CardDetector
import org.openscanvision.core.internal.omr.Templates
import java.nio.ByteBuffer
import kotlin.math.sqrt

private const val TAG = "TrackingUtils"

// ─── Constants ──────────────────────────────────────────────────────────
const val MAX_FRAMES_BEFORE_RESCAN = 120
const val LOW_CONF_FRAMES_BEFORE_RESCAN = 10
const val BASE_TRACK_HALF_SIZE = 55
const val MIN_TRACK_HALF_SIZE = 30
const val MAX_TRACK_HALF_SIZE = 100
const val CONFIDENCE_THRESHOLD = 5
const val CAPTURE_CONFIDENCE_THRESHOLD = 1
const val MIN_CAPTURE_HIGH_CONF_MARKERS = 3
const val REQUIRED_MARKERS_FOR_CAPTURE = 4
const val PREDICTED_MARKER_SEARCH_HALF_SIZE = 90
const val MIN_CAPTURE_AVG_CONF = 1f
const val MAX_CAPTURE_HOMOGRAPHY_ERROR_PX = 12f
const val MAX_MARKER_AREA_RATIO = 3.0f
const val CONFIDENCE_DECAY = 0.92f
const val PERSISTENCE_FRAMES = 20
const val ACCELERATION_NOISE = 0.02f
const val AUTO_CAPTURE_STABLE_FRAMES = 1
const val AUTO_CAPTURE_COOLDOWN_FRAMES = 4
const val QUICK_RETRY_DELAY_FRAMES = 2

// ─── Kalman filter ──────────────────────────────────────────────────
data class KalmanState(
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

// ─── Tracking state holder ──────────────────────────────────────────
class TrackingState {
    var frameCounter = 0
    val kalmanFilters = mutableMapOf<Int, KalmanState>()
    val markerConfidence = mutableMapOf<Int, Int>()
    val markerAge = mutableMapOf<Int, Int>()
    var lastKnownCentres: Map<Int, PointF> = emptyMap()
    var previousCentres: Map<Int, PointF> = emptyMap()
    var framesSinceFullScan = MAX_FRAMES_BEFORE_RESCAN
    var stableFrameCounter = 0
    var autoCaptureCooldown = 0
    var quickRetryDelay = 0
    var retryPending = false
    var lockStartFrame = -1
    var lockStartTimeMs = 0L
    var metricsAttempts = 0
    var metricsSuccesses = 0
    var metricsQualityRejects = 0
    var metricsTotalLockFrames = 0L
    var metricsTotalTimeToCaptureMs = 0L
}

data class TrackingResult(
    val arUcoMap: Map<Int, List<PointF>>,
    val isTracking: Boolean,
    val isStable: Boolean
)

// ─── Helper functions ──────────────────────────────────────────────

fun yPlaneToGrayMat(imageProxy: ImageProxy): Mat {
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

fun centreOfCorners(corners: List<PointF>): PointF {
    var x = 0f; var y = 0f
    for (c in corners) { x += c.x; y += c.y }
    val n = corners.size.coerceAtLeast(1).toFloat()
    return PointF(x / n, y / n)
}

fun cardCornersFromArUco(arUcoMap: Map<Int, List<PointF>>): List<PointF>? {
    val homography = CardDetector.buildHomographyFromArUco(arUcoMap) ?: return null
    val templateCardCorners = listOf(
        PointF(0f, 0f),
        PointF((Templates.REF_WIDTH - 1).toFloat(), 0f),
        PointF((Templates.REF_WIDTH - 1).toFloat(), (Templates.REF_HEIGHT - 1).toFloat()),
        PointF(0f, (Templates.REF_HEIGHT - 1).toFloat())
    )
    return CardDetector.predictImagePoints(templateCardCorners, homography)
}

fun markerArea(corners: List<PointF>): Float {
    if (corners.size != 4) return 0f
    var area = 0f
    for (i in corners.indices) {
        val next = corners[(i + 1) % corners.size]
        area += corners[i].x * next.y - next.x * corners[i].y
    }
    return kotlin.math.abs(area) * 0.5f
}

fun validateCaptureQuality(
    arUcoMap: Map<Int, List<PointF>>, markerConfidence: Map<Int, Int>
): Pair<Boolean, String> {
    val highConfMarkers = arUcoMap.keys.count { (markerConfidence[it] ?: 0) >= CAPTURE_CONFIDENCE_THRESHOLD }
    if (highConfMarkers < REQUIRED_MARKERS_FOR_CAPTURE)
        return Pair(false, "Need all 4 markers stable (have $highConfMarkers).")
    val avgConfidence = arUcoMap.keys.map { markerConfidence[it] ?: 0 }.average().toFloat()
    if (avgConfidence < MIN_CAPTURE_AVG_CONF)
        return Pair(false, "Marker confidence too low.")
    val areas = arUcoMap.values.map(::markerArea).filter { it > 1f }
    if (areas.size >= 2) {
        val minArea = areas.minOrNull() ?: 0f
        val maxArea = areas.maxOrNull() ?: 0f
        if (minArea <= 0f || maxArea / minArea > MAX_MARKER_AREA_RATIO)
            return Pair(false, "Marker consistency check failed.")
    }
    val homographyError = CardDetector.computeArUcoHomographyError(arUcoMap)
    if (homographyError == null || homographyError > MAX_CAPTURE_HOMOGRAPHY_ERROR_PX)
        return Pair(false, "Homography error too high.")
    return Pair(true, "")
}

// ─── Main tracking update (no overlay) ──────────────────────────────

fun updateTracking(
    gray: Mat,
    trackingState: TrackingState,
    captureRequested: () -> Boolean = { false }
): TrackingResult {
    val state = trackingState
    val lastCentres = state.lastKnownCentres
    val prevCentres = state.previousCentres
    var arUcoMap: Map<Int, List<PointF>> = emptyMap()

    try {
        val capturePending = captureRequested()
        val highConfidenceTracked = state.markerConfidence.count { it.value >= CONFIDENCE_THRESHOLD }
        val lowConfidenceTracking = highConfidenceTracked < MIN_CAPTURE_HIGH_CONF_MARKERS
        val shouldTryTracking = lastCentres.isNotEmpty()
        if (shouldTryTracking) {
            val allActiveIds = lastCentres.keys + state.markerAge.filter { it.value < PERSISTENCE_FRAMES }.keys
            val predictedCentres = allActiveIds.associateWith { id ->
                val kalman = state.kalmanFilters[id]
                if (kalman != null) { kalman.predict(); PointF(kalman.x, kalman.y) }
                else lastCentres[id] ?: prevCentres[id] ?: PointF(0f, 0f)
            }
            val halfSizeMap = allActiveIds.associateWith { id ->
                val conf = state.markerConfidence[id] ?: 0
                when {
                    conf >= CONFIDENCE_THRESHOLD -> MIN_TRACK_HALF_SIZE
                    conf >= 2 -> BASE_TRACK_HALF_SIZE
                    else -> MAX_TRACK_HALF_SIZE
                }
            }
            val tracked = CardDetector.detectArUcoMarkersTrackedInGray(gray, predictedCentres, halfSizeMap)
            val templatePoints = Templates.SHARED_MARKER_CENTRES
            val filtered = if (tracked.size >= 4) CardDetector.rejectOutliersWithHomography(tracked, templatePoints) else tracked
            if (filtered.size >= allActiveIds.size / 2) state.framesSinceFullScan++ else state.framesSinceFullScan = (state.framesSinceFullScan + LOW_CONF_FRAMES_BEFORE_RESCAN).coerceAtMost(MAX_FRAMES_BEFORE_RESCAN)
            arUcoMap = filtered
        }
        val shouldReacquire = arUcoMap.size < MIN_CAPTURE_HIGH_CONF_MARKERS && (!shouldTryTracking || capturePending || (lowConfidenceTracking && state.framesSinceFullScan >= LOW_CONF_FRAMES_BEFORE_RESCAN) || state.framesSinceFullScan >= MAX_FRAMES_BEFORE_RESCAN)
        if (shouldReacquire) {
            val reacquired = CardDetector.detectArUcoMarkersReacquire(gray, listOf(0.4, 1.0), earlyBreakCount = 3)
            val templatePoints = Templates.SHARED_MARKER_CENTRES
            arUcoMap = if (reacquired.size >= MIN_CAPTURE_HIGH_CONF_MARKERS) CardDetector.rejectOutliersWithHomography(reacquired, templatePoints) else reacquired
            state.framesSinceFullScan = 0
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
    } catch (e: Exception) {
        Log.e(TAG, "Tracking error", e)
    }

    // Update confidence and age
    val detectedIds = arUcoMap.keys
    for (id in state.markerAge.keys) state.markerAge[id] = (state.markerAge[id] ?: 0) + 1
    for (id in detectedIds) {
        val corners = arUcoMap[id] ?: continue
        var cx = 0f; var cy = 0f
        for (corner in corners) { cx += corner.x; cy += corner.y }
        val count = corners.size.coerceAtLeast(1).toFloat()
        cx /= count; cy /= count
        val kalman = state.kalmanFilters[id]
        if (kalman != null) kalman.update(cx, cy) else state.kalmanFilters[id] = KalmanState(cx, cy, 0f, 0f, 0f, 0f)
        state.markerConfidence[id] = (state.markerConfidence[id] ?: 0) + 1
        state.markerAge[id] = 0
    }
    for (id in state.markerConfidence.keys) {
        if (id !in detectedIds) state.markerConfidence[id] = ((state.markerConfidence[id] ?: 0) * CONFIDENCE_DECAY).toInt().coerceAtLeast(1)
    }
    val toRemove = state.markerAge.filter { it.value > PERSISTENCE_FRAMES && (state.markerConfidence[it.key] ?: 0) < 3 }.keys
    for (id in toRemove) {
        state.kalmanFilters.remove(id); state.markerConfidence.remove(id); state.markerAge.remove(id)
    }
    state.previousCentres = state.lastKnownCentres
    state.lastKnownCentres = arUcoMap.mapValues { (_, corners) ->
        var x = 0f; var y = 0f
        for (corner in corners) { x += corner.x; y += corner.y }
        val count = corners.size.coerceAtLeast(1).toFloat()
        PointF(x / count, y / count)
    }

    val highConfDetected = detectedIds.count { (state.markerConfidence[it] ?: 0) >= CAPTURE_CONFIDENCE_THRESHOLD }
    val isStable = highConfDetected >= REQUIRED_MARKERS_FOR_CAPTURE && arUcoMap.size >= REQUIRED_MARKERS_FOR_CAPTURE

    if (isStable) {
        state.stableFrameCounter++
        if (state.stableFrameCounter == 1) {
            state.lockStartFrame = state.frameCounter
            state.lockStartTimeMs = System.currentTimeMillis()
        }
    } else {
        state.stableFrameCounter = 0
        state.lockStartFrame = -1
        state.lockStartTimeMs = 0L
    }

    if (state.autoCaptureCooldown > 0) state.autoCaptureCooldown--
    if (state.quickRetryDelay > 0) state.quickRetryDelay--

    return TrackingResult(arUcoMap, arUcoMap.isNotEmpty(), isStable)
}