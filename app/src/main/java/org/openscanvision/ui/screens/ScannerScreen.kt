package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.PointF
import android.net.Uri
import android.os.Environment
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
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.openscanvision.omr.CardDetector
import org.openscanvision.omr.Templates
import org.openscanvision.ui.components.StaticViewfinder
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

private const val TAG = "ArUcoScanner"

// ─── Analysis settings ───────────────────────────────────────────
private val ANALYSIS_TARGET_RESOLUTION = android.util.Size(640, 480)

// ─── Tracking speed / recovery ──────────────────────────────────
private const val MAX_FRAMES_BEFORE_RESCAN = 8
private const val TRACK_HALF_SIZE_PX = 70
private const val REACQUIRE_DOWNSCALE = 0.5
private const val DISPLAY_SMOOTHING_ALPHA = 0.55f
private val RECOVERY_STAGES = intArrayOf(110, 220)   // search radii (pixels) — trimmed from 3 stages to 2 for speed

// ─── Template matching fallback ──────────────────────────────────
private const val TEMPLATE_SIZE = 64
private const val MATCH_THRESHOLD = 0.6

// ─── Auto‑capture ─────────────────────────────────────────────────
// Capture fires as soon as all 4 markers are seen with genuine (non-template-matched)
// detections for CONFIRM_FRAMES consecutive frames — just enough to avoid a one-frame
// flicker/false-positive, without waiting for a steady hand-hold like before.
private const val REQUIRED_MARKER_COUNT = 4
private const val CONFIRM_FRAMES = 2

// ─── Geometric cross-check tolerances ────────────────────────────
// Used to sanity-check detected markers against each other (and the fixed, known card
// layout) before trusting them to predict where the remaining markers should be.
private const val PAIR_SCALE_MIN = 0.15f
private const val PAIR_SCALE_MAX = 8.0f
private const val SCALE_AGREEMENT_TOLERANCE = 0.30f   // ±30% vs the median pairwise scale
private const val ANGLE_AGREEMENT_TOLERANCE_DEG = 18f // ±18° vs the median pairwise rotation

// ─── Display persistence (sticky markers) ─────────────────────────
private const val PERSISTENCE_WINDOW = 5

// ─── Helper functions (kept from your original) ─────────────────

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
            for (col in 0 until width) {
                outRow[col] = rowBytes[col * pixelStride]
            }
            mat.put(row, 0, outRow)
        }
    }
    return mat
}

/** Light unsharp mask to sharpen marker edges before detection */
private fun enhanceMat(src: Mat): Mat {
    val blurred = Mat()
    Imgproc.GaussianBlur(src, blurred, Size(0.0, 0.0), 3.0)
    val dst = Mat()
    Core.addWeighted(src, 1.5, blurred, -0.5, 0.0, dst)
    blurred.release()
    return dst
}

/** Create a timestamped output file for high‑res capture */
private fun createOutputFile(context: android.content.Context): File {
    val dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: context.filesDir
    if (!dir.exists()) dir.mkdirs()
    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
    return File(dir, "ballot_$timestamp.jpg")
}

// ─── Model‑based prediction & recovery ──────────────────────────

/** Marker ID -> position. */
typealias CardModel = Map<Int, PointF>

/**
 * The physical marker layout (TL=0, TR=1, BR=2, BL=3) is fixed by the card template
 * itself — it's the same for every card, so we don't need to "learn" it from a full
 * detection first. It's available from frame one, even with only a single marker visible.
 */
private val FIXED_CARD_MODEL: CardModel =
    Templates.SHARED_MARKER_CENTRES.withIndex().associate { (id, p) -> id to p }

private fun applySimilarity(modelPos: PointF, transform: FloatArray): PointF {
    val (angle, scale, tx, ty) = transform
    return PointF(
        scale * (cos(angle) * modelPos.x - sin(angle) * modelPos.y) + tx,
        scale * (sin(angle) * modelPos.x + cos(angle) * modelPos.y) + ty
    )
}

/**
 * Best-fit similarity transform (rotation + uniform scale + translation, no reflection)
 * from model points to image points, using ALL given correspondences at once — a closed-form
 * least-squares fit (equivalent to combining every pairwise estimate), not just one arbitrary
 * pair. With exactly 2 points this reduces to the direct two-point estimate; with 3+ it's a
 * genuine best fit across all of them.
 */
private fun estimateSimilarityLS(model: CardModel, imageCentres: Map<Int, PointF>): FloatArray? {
    val ids = model.keys.intersect(imageCentres.keys).toList()
    if (ids.size < 2) return null

    val mcx = ids.map { model[it]!!.x }.average().toFloat()
    val mcy = ids.map { model[it]!!.y }.average().toFloat()
    val icx = ids.map { imageCentres[it]!!.x }.average().toFloat()
    val icy = ids.map { imageCentres[it]!!.y }.average().toFloat()

    var numReal = 0.0; var numImag = 0.0; var denom = 0.0
    for (id in ids) {
        val mx = (model[id]!!.x - mcx).toDouble(); val my = (model[id]!!.y - mcy).toDouble()
        val ix = (imageCentres[id]!!.x - icx).toDouble(); val iy = (imageCentres[id]!!.y - icy).toDouble()
        numReal += mx * ix + my * iy
        numImag += mx * iy - my * ix
        denom += mx * mx + my * my
    }
    if (denom < 1e-6) return null
    val cReal = (numReal / denom).toFloat()
    val cImag = (numImag / denom).toFloat()
    val scale = hypot(cReal.toDouble(), cImag.toDouble()).toFloat()
    if (scale < 1e-4) return null
    val rotation = atan2(cImag, cReal)
    val tx = icx - (cReal * mcx - cImag * mcy)
    val ty = icy - (cImag * mcx + cReal * mcy)
    return floatArrayOf(rotation, scale, tx, ty)
}

/**
 * Cross-checks detected markers against the known fixed layout and against each other,
 * dropping anything that doesn't fit before it's allowed to influence a prediction:
 *  - 1 marker: nothing to cross-check against — passed through as-is.
 *  - 2 markers: sanity-bound the implied scale (catches a wildly wrong correspondence).
 *  - 3+ markers: compare every pair's implied scale/rotation against the group median;
 *    a marker that's only ever part of disagreeing pairs is dropped as a likely
 *    false/misread detection ("combination of them" cross-check).
 */
private fun verifyAgainstModel(model: CardModel, imageCentres: Map<Int, PointF>): Map<Int, PointF> {
    if (imageCentres.size < 2) return imageCentres

    data class PairEstimate(val idA: Int, val idB: Int, val scale: Float, val angleDeg: Float)

    val ids = imageCentres.keys.toList()
    val estimates = mutableListOf<PairEstimate>()
    for (i in ids.indices) {
        for (j in i + 1 until ids.size) {
            val id1 = ids[i]; val id2 = ids[j]
            val m1 = model[id1] ?: continue; val m2 = model[id2] ?: continue
            val modelDist = hypot((m2.x - m1.x).toDouble(), (m2.y - m1.y).toDouble()).toFloat()
            if (modelDist < 1e-3f) continue
            val p1 = imageCentres[id1]!!; val p2 = imageCentres[id2]!!
            val dx = p2.x - p1.x; val dy = p2.y - p1.y
            val imgDist = hypot(dx.toDouble(), dy.toDouble()).toFloat()
            if (imgDist < 1e-3f) continue
            val scale = imgDist / modelDist
            val modelAngle = atan2((m2.y - m1.y).toDouble(), (m2.x - m1.x).toDouble())
            var angleDiffDeg = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble()) - modelAngle).toFloat()
            angleDiffDeg = ((angleDiffDeg + 180f) % 360f + 360f) % 360f - 180f
            estimates.add(PairEstimate(id1, id2, scale, angleDiffDeg))
        }
    }
    if (estimates.isEmpty()) return imageCentres

    if (imageCentres.size == 2) {
        val est = estimates.first()
        return if (est.scale in PAIR_SCALE_MIN..PAIR_SCALE_MAX) imageCentres else {
            Log.w(TAG, "Rejecting 2-marker pair (${est.idA},${est.idB}): implausible scale ${est.scale}")
            emptyMap()
        }
    }

    // 3+ markers: majority-vote outlier rejection across all pairwise combinations.
    val medianScale = estimates.map { it.scale }.sorted()[estimates.size / 2]
    val medianAngle = estimates.map { it.angleDeg }.sorted()[estimates.size / 2]
    val agree = mutableMapOf<Int, Int>(); val disagree = mutableMapOf<Int, Int>()
    for (est in estimates) {
        val ok = est.scale in PAIR_SCALE_MIN..PAIR_SCALE_MAX &&
                abs(est.scale - medianScale) <= medianScale * SCALE_AGREEMENT_TOLERANCE &&
                abs(est.angleDeg - medianAngle) <= ANGLE_AGREEMENT_TOLERANCE_DEG
        for (id in listOf(est.idA, est.idB)) {
            if (ok) agree[id] = (agree[id] ?: 0) + 1 else disagree[id] = (disagree[id] ?: 0) + 1
        }
    }
    val verified = imageCentres.filterKeys { id -> (agree[id] ?: 0) >= (disagree[id] ?: 0) }
    if (verified.size < imageCentres.size) {
        Log.w(TAG, "Dropped geometrically inconsistent marker(s): ${imageCentres.keys - verified.keys}")
    }
    return verified
}

/**
 * Staged prediction for markers not yet found this frame, given the ones that are:
 *  - 1 verified marker → translate the fixed layout using it, reusing the last known
 *    rotation/scale if we have one from a previous confident frame (otherwise assumes
 *    upright/unscaled) — enough to predict roughly where the other 3 should be.
 *  - 2+ verified markers → a proper least-squares transform from all of them.
 * Either way, the result is only ever used to aim a local search — see recoverMarkerRobust —
 * never treated as a confirmed detection on its own.
 */
private fun predictMissingMarkers(
    model: CardModel,
    verifiedCentres: Map<Int, PointF>,
    lastTransform: FloatArray?
): Pair<Map<Int, PointF>, FloatArray?> {
    if (verifiedCentres.isEmpty()) return Pair(emptyMap(), null)

    val transform = if (verifiedCentres.size >= 2) estimateSimilarityLS(model, verifiedCentres) else null
    val predictions = mutableMapOf<Int, PointF>()

    val effective = transform ?: lastTransform
    if (effective != null) {
        for ((id, modelPos) in model) {
            if (id !in verifiedCentres) predictions[id] = applySimilarity(modelPos, effective)
        }
        return Pair(predictions, transform)
    }

    // No transform available at all yet (first-ever frame with just 1 marker): fall back
    // to a pure translation assuming the card is roughly upright and at template scale.
    val refId = verifiedCentres.keys.first()
    val refModelPos = model[refId] ?: return Pair(emptyMap(), null)
    val refImgPos = verifiedCentres[refId]!!
    val dx = refImgPos.x - refModelPos.x
    val dy = refImgPos.y - refModelPos.y
    for ((id, modelPos) in model) {
        if (id !in verifiedCentres) predictions[id] = PointF(modelPos.x + dx, modelPos.y + dy)
    }
    return Pair(predictions, null)
}

/**
 * Recovers a single lost marker around a predicted centre.
 * Returns Triple(id, corners, isTemplateMatch). isTemplateMatch=false means a genuine
 * ArUco detection (trustworthy enough to count toward an instant capture); true means
 * it was only located via correlation template matching (positional guess — good enough
 * to keep tracking/UI smooth, but not enough on its own to trigger the high-res capture).
 */
private fun recoverMarkerRobust(
    gray: Mat,
    predictedCentre: PointF,
    markerId: Int,
    lastCorners: List<PointF>?,
    templateCache: MutableMap<Int, Mat>
): Triple<Int, List<PointF>, Boolean>? {
    for (halfSize in RECOVERY_STAGES) {
        val left = (predictedCentre.x - halfSize).toInt().coerceIn(0, gray.cols() - 1)
        val top = (predictedCentre.y - halfSize).toInt().coerceIn(0, gray.rows() - 1)
        val right = (predictedCentre.x + halfSize).toInt().coerceIn(0, gray.cols() - 1)
        val bottom = (predictedCentre.y + halfSize).toInt().coerceIn(0, gray.rows() - 1)
        if (right <= left || bottom <= top) continue
        val roi = gray.submat(top, bottom, left, right)
        val enhancedRoi = enhanceMat(roi)
        val detections = CardDetector.detectArUcoMarkersReacquire(enhancedRoi, 1.0)
        enhancedRoi.release()
        roi.release()
        for ((id, corners) in detections) {
            val cx = corners.map { it.x }.average().toFloat() + left
            val cy = corners.map { it.y }.average().toFloat() + top
            val dist = hypot((cx - predictedCentre.x).toDouble(), (cy - predictedCentre.y).toDouble())
            if (dist < halfSize * 0.7) {
                return Triple(id, corners.map { PointF(it.x + left, it.y + top) }, false)
            }
        }
    }

    // Template matching (lower-confidence positional fallback)
    val template = templateCache[markerId] ?: return null
    if (template.empty()) return null
    val half = RECOVERY_STAGES.last()
    val left = (predictedCentre.x - half).toInt().coerceIn(0, gray.cols() - 1)
    val top = (predictedCentre.y - half).toInt().coerceIn(0, gray.rows() - 1)
    val right = (predictedCentre.x + half).toInt().coerceIn(0, gray.cols() - 1)
    val bottom = (predictedCentre.y + half).toInt().coerceIn(0, gray.rows() - 1)
    if (right <= left || bottom <= top) return null
    val searchRegion = gray.submat(top, bottom, left, right)
    val result = Mat()
    Imgproc.matchTemplate(searchRegion, template, result, Imgproc.TM_CCOEFF_NORMED)
    val minMax = Core.minMaxLoc(result)
    result.release(); searchRegion.release()
    if (minMax.maxVal >= MATCH_THRESHOLD) {
        val matchX = minMax.maxLoc.x + left + template.cols() / 2.0
        val matchY = minMax.maxLoc.y + top + template.rows() / 2.0
        val corners = if (lastCorners != null && lastCorners.size == 4) {
            val avgCx = lastCorners.map { it.x }.average().toFloat()
            val avgCy = lastCorners.map { it.y }.average().toFloat()
            lastCorners.map { PointF(it.x + (matchX - avgCx).toFloat(), it.y + (matchY - avgCy).toFloat()) }
        } else {
            val r = 20f
            listOf(PointF(matchX.toFloat()-r, matchY.toFloat()-r), PointF(matchX.toFloat()+r, matchY.toFloat()-r),
                PointF(matchX.toFloat()+r, matchY.toFloat()+r), PointF(matchX.toFloat()-r, matchY.toFloat()+r))
        }
        return Triple(markerId, corners, true)
    }
    return null
}

/** Save a reference patch for a marker when first fully detected. */
private fun updateTemplate(gray: Mat, corners: List<PointF>, markerId: Int, cache: MutableMap<Int, Mat>) {
    if (cache.containsKey(markerId)) return
    val points = corners.map { Point(it.x.toDouble(), it.y.toDouble()) }
    val contour = MatOfPoint2f()
    contour.fromList(points)
    val rect = Imgproc.boundingRect(contour)
    contour.release()
    if (rect.width <= 0 || rect.height <= 0) return
    val roi = gray.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width)
    val resized = Mat()
    Imgproc.resize(roi, resized, Size(TEMPLATE_SIZE.toDouble(), TEMPLATE_SIZE.toDouble()))
    cache[markerId] = resized.clone()
    roi.release(); resized.release()
}

// ─── Composable ─────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen(onCardCaptured: ((Uri) -> Unit)? = null) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasCameraPermission = it }

    var detectedMarkers by remember { mutableStateOf<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) }
    var isTracking by remember { mutableStateOf(false) }
    var stableFrameCount by remember { mutableStateOf(0) }
    var captureState by remember { mutableStateOf<CaptureState>(CaptureState.Scanning) }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewLaidOut by remember { mutableStateOf(false) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }

    // Thread‑safe atoms
    val lastKnownCentresRef = remember { AtomicReference<Map<Int, PointF>>(emptyMap()) }
    val previousCentresRef = remember { AtomicReference<Map<Int, PointF>>(emptyMap()) }
    val lastKnownCornersRef = remember { AtomicReference<Map<Int, List<PointF>>>(emptyMap()) }
    val smoothedMarkersRef = remember { AtomicReference<Map<Int, Pair<Offset, List<Offset>>>>(emptyMap()) }
    val framesSinceFullScanRef = remember { AtomicInteger(MAX_FRAMES_BEFORE_RESCAN) }
    val stableFrameCounterRef = remember { AtomicInteger(0) }
    val captureTriggeredRef = remember { AtomicBoolean(false) }
    val templateCacheRef = remember { AtomicReference<MutableMap<Int, Mat>>(HashMap()) }
    val lastTransformRef = remember { AtomicReference<FloatArray?>(null) } // [angle, scale, tx, ty] from the last confidently-fit frame

    var presenceWindow by remember { mutableStateOf<Map<Int, List<Boolean>>>(emptyMap()) }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            captureExecutor.shutdown()
            templateCacheRef.get().values.forEach { it.release() }
        }
    }

    DisposableEffect(previewView) {
        val listener = android.view.View.OnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            if (!previewLaidOut && previewView.width > 0 && previewView.height > 0)
                previewLaidOut = true
        }
        previewView.addOnLayoutChangeListener(listener)
        onDispose { previewView.removeOnLayoutChangeListener(listener) }
    }

    fun resetScan() {
        captureTriggeredRef.set(false)
        stableFrameCounterRef.set(0)
        lastKnownCentresRef.set(emptyMap())
        previousCentresRef.set(emptyMap())
        lastKnownCornersRef.set(emptyMap())
        framesSinceFullScanRef.set(MAX_FRAMES_BEFORE_RESCAN)
        stableFrameCount = 0
        captureState = CaptureState.Scanning
        presenceWindow = emptyMap()
        // Fresh card each scan: don't let stale rotation/scale or templates from the
        // last card bias detection of the next one.
        lastTransformRef.set(null)
        templateCacheRef.get().values.forEach { it.release() }
        templateCacheRef.set(HashMap())
    }

    fun captureHighRes() {
        val capture = imageCapture ?: return
        captureState = CaptureState.Capturing
        val outputFile = createOutputFile(context)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()
        capture.takePicture(
            outputOptions, captureExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val uri = output.savedUri ?: Uri.fromFile(outputFile)
                    Log.d(TAG, "High-res capture saved: $uri")
                    coroutineScope.launch(Dispatchers.Main) {
                        captureState = CaptureState.Captured(uri)
                        onCardCaptured?.invoke(uri)
                    }
                }
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "High-res capture failed", exception)
                    coroutineScope.launch(Dispatchers.Main) {
                        captureState = CaptureState.Error(exception.message ?: "Capture failed")
                        captureTriggeredRef.set(false)
                        stableFrameCounterRef.set(0)
                    }
                }
            }
        )
    }

    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()

                val preview = Preview.Builder().build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(ANALYSIS_TARGET_RESOLUTION)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                val capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .build()
                imageCapture = capture

                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    val mediaImage = imageProxy.image
                    if (mediaImage == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    if (captureTriggeredRef.get()) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val bufferWidth = imageProxy.width.toFloat()
                    val bufferHeight = imageProxy.height.toFloat()
                    val rotatedWidth = if (rotationDegrees % 180 == 0) bufferWidth else bufferHeight
                    val rotatedHeight = if (rotationDegrees % 180 == 0) bufferHeight else bufferWidth

                    val gray = yPlaneToGrayMat(imageProxy)
                    imageProxy.close()

                    try {
                        val enhanced = enhanceMat(gray)

                        val lastCentres = lastKnownCentresRef.get()
                        val prevCentres = previousCentresRef.get()
                        val lastCorners = lastKnownCornersRef.get()
                        val needsReacquire = lastCentres.isEmpty() || framesSinceFullScanRef.get() >= MAX_FRAMES_BEFORE_RESCAN

                        // ids resolved this frame only via low-confidence template matching —
                        // fine for smooth tracking/UI, but excluded from the capture-trigger check.
                        val templateMatchedIds = mutableSetOf<Int>()

                        // 1. Primary detection
                        var arUcoMap: Map<Int, List<PointF>> = if (!needsReacquire) {
                            val predictedCentres = lastCentres.mapValues { (id, centre) ->
                                val prev = prevCentres[id]
                                if (prev != null) PointF(centre.x + (centre.x - prev.x), centre.y + (centre.y - prev.y))
                                else centre
                            }
                            var tracked = CardDetector.detectArUcoMarkersTrackedInGray(enhanced, predictedCentres, TRACK_HALF_SIZE_PX)
                            if (tracked.size < lastCentres.size) {
                                val finalTracked = tracked.toMutableMap()
                                val templateCache = templateCacheRef.get()
                                for ((id, lastCentre) in lastCentres) {
                                    if (!tracked.containsKey(id)) {
                                        val recovered = recoverMarkerRobust(enhanced, lastCentre, id, lastCorners[id], templateCache)
                                        if (recovered != null) {
                                            finalTracked[id] = recovered.second
                                            if (recovered.third) templateMatchedIds.add(id)
                                        }
                                    }
                                }
                                tracked = finalTracked
                                framesSinceFullScanRef.set(if (tracked.size >= lastCentres.size) framesSinceFullScanRef.get() + 1 else MAX_FRAMES_BEFORE_RESCAN)
                            } else {
                                framesSinceFullScanRef.incrementAndGet()
                            }
                            tracked
                        } else {
                            val reacquired = CardDetector.detectArUcoMarkersReacquire(enhanced, REACQUIRE_DOWNSCALE)
                            val templateCache = templateCacheRef.get()
                            for ((id, corners) in reacquired) updateTemplate(enhanced, corners, id, templateCache)
                            framesSinceFullScanRef.set(0)
                            reacquired
                        }

                        // 2. Geometric cross-check + staged prediction for still-missing markers.
                        //    1 found  -> predict the other 3 from the fixed layout (+ last known
                        //                rotation/scale if we have it).
                        //    2 found  -> sanity-check the pair, then fit rotation+scale+translation
                        //                from them and predict the other 2.
                        //    3 found  -> cross-check all 3 pairwise combinations (drops an outlier
                        //                if one disagrees), fit from what's left, predict the 4th.
                        //    4 found  -> nothing to predict; falls straight through to capture below.
                        if (arUcoMap.size < REQUIRED_MARKER_COUNT) {
                            val rawCentres = arUcoMap.mapValues { (_, c) ->
                                PointF(c.map { it.x }.average().toFloat(), c.map { it.y }.average().toFloat())
                            }
                            val verifiedCentres = verifyAgainstModel(FIXED_CARD_MODEL, rawCentres)
                            val rejectedIds = rawCentres.keys - verifiedCentres.keys
                            if (rejectedIds.isNotEmpty()) {
                                // Treat as not-found this frame rather than let a geometrically
                                // inconsistent (likely false) detection pollute the prediction
                                // or the capture-trigger check.
                                arUcoMap = arUcoMap - rejectedIds
                                templateMatchedIds.removeAll(rejectedIds)
                            }

                            val (predictions, newTransform) =
                                predictMissingMarkers(FIXED_CARD_MODEL, verifiedCentres, lastTransformRef.get())
                            if (newTransform != null) lastTransformRef.set(newTransform)

                            val templateCache = templateCacheRef.get()
                            for ((missingId, predCentre) in predictions) {
                                val recovered = recoverMarkerRobust(enhanced, predCentre, missingId, lastCorners[missingId], templateCache)
                                if (recovered != null) {
                                    arUcoMap = arUcoMap + (recovered.first to recovered.second)
                                    if (recovered.third) templateMatchedIds.add(recovered.first)
                                }
                            }
                        }

                        enhanced.release()

                        // Instant-capture gate: fire on the first CONFIRM_FRAMES consecutive
                        // frames where all 4 markers are seen AND none of them are just a
                        // template-match guess. No steady-hold wait — as soon as a genuinely
                        // confident 4-marker read lands, we go straight to the high-res capture.
                        val newCentres = arUcoMap.mapValues { (_, c) ->
                            PointF(c.map { it.x }.average().toFloat(), c.map { it.y }.average().toFloat())
                        }
                        val hasFullSet = newCentres.size >= REQUIRED_MARKER_COUNT
                        val allConfident = hasFullSet && arUcoMap.keys.none { it in templateMatchedIds }
                        val currentStable = if (allConfident) stableFrameCounterRef.incrementAndGet() else stableFrameCounterRef.updateAndGet { 0 }
                        val shouldCapture = !captureTriggeredRef.get() && currentStable >= CONFIRM_FRAMES
                        if (shouldCapture) captureTriggeredRef.set(true)

                        previousCentresRef.set(lastCentres)
                        lastKnownCentresRef.set(newCentres)
                        lastKnownCornersRef.set(arUcoMap)

                        // Screen mapping
                        val rawScreenMarkers = mutableMapOf<Int, Pair<Offset, List<Offset>>>()
                        for ((id, corners) in arUcoMap) {
                            val sensorCorners = mapBitmapPointsToSensor(corners, bufferWidth, bufferHeight, rotationDegrees)
                            val screenCorners = mapImageToScreen(
                                sensorCorners.map { Offset(it.x, it.y) }, rotatedWidth, rotatedHeight, previewView
                            )
                            if (screenCorners != null && screenCorners.size == 4) {
                                val cx = screenCorners.map { it.x }.average().toFloat()
                                val cy = screenCorners.map { it.y }.average().toFloat()
                                rawScreenMarkers[id] = Pair(Offset(cx, cy), screenCorners)
                            }
                        }

                        val prevSmoothed = smoothedMarkersRef.get()
                        val smoothed = rawScreenMarkers.mapValues { (id, raw) ->
                            val prev = prevSmoothed[id]
                            if (prev == null) raw
                            else {
                                val (rawCentre, rawCorners) = raw
                                val (prevCentre, prevCorners) = prev
                                val a = DISPLAY_SMOOTHING_ALPHA
                                val newCentre = Offset(
                                    prevCentre.x + a * (rawCentre.x - prevCentre.x),
                                    prevCentre.y + a * (rawCentre.y - prevCentre.y))
                                val newCorners = rawCorners.indices.map { i ->
                                    Offset(prevCorners[i].x + a * (rawCorners[i].x - prevCorners[i].x),
                                        prevCorners[i].y + a * (rawCorners[i].y - prevCorners[i].y))
                                }
                                Pair(newCentre, newCorners)
                            }
                        }
                        smoothedMarkersRef.set(smoothed)

                        coroutineScope.launch(Dispatchers.Main) {
                            // Sticky persistence
                            val updatedWindow = presenceWindow.toMutableMap()
                            val currentIds = smoothed.keys
                            for (id in (updatedWindow.keys + currentIds).distinct()) {
                                val history = (updatedWindow[id] ?: listOf()).takeLast(PERSISTENCE_WINDOW - 1) + (id in currentIds)
                                updatedWindow[id] = history
                            }
                            updatedWindow.entries.removeAll { it.value.all { !it } }
                            presenceWindow = updatedWindow

                            val finalDisplay = mutableMapOf<Int, Pair<Offset, List<Offset>>>()
                            for ((id, history) in presenceWindow) {
                                if (history.any { it }) {
                                    finalDisplay[id] = smoothed[id] ?: detectedMarkers[id] ?: continue
                                }
                            }
                            detectedMarkers = finalDisplay
                            isTracking = finalDisplay.isNotEmpty()
                            stableFrameCount = currentStable.coerceAtMost(CONFIRM_FRAMES)

                            if (shouldCapture) captureHighRes()
                        }
                    } finally {
                        gray.release()
                    }
                }

                // ViewPort binding
                val existingViewPort = previewView.viewPort
                val viewPort = existingViewPort ?: run {
                    val w = previewView.width; val h = previewView.height
                    if (w > 0 && h > 0)
                        ViewPort.Builder(Rational(w, h), previewView.display?.rotation ?: android.view.Surface.ROTATION_0).build()
                    else null
                }
                val useCaseGroup = UseCaseGroup.Builder()
                    .addUseCase(preview).addUseCase(analysis).addUseCase(capture)
                    .apply { if (viewPort != null) setViewPort(viewPort) }
                    .build()
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, useCaseGroup)

                cameraError = null
                Log.d(TAG, "Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera start error", e)
                cameraError = "Failed to start camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }

    LaunchedEffect(hasCameraPermission, previewLaidOut) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
        else if (previewLaidOut) startCamera()
    }

    // Overlay (unchanged)
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
                drawPath(path, Color.Green, style = Stroke(width = 4f))
                drawCircle(Color.Red, radius = 8f, center = centre)
                drawContext.canvas.nativeCanvas.apply {
                    val paint = android.graphics.Paint().apply {
                        color = android.graphics.Color.WHITE; textSize = 40f; isAntiAlias = true
                        setShadowLayer(4f, 2f, 2f, android.graphics.Color.BLACK)
                    }
                    drawText("ID: $id  X: ${centre.x.toInt()}  Y: ${centre.y.toInt()}", centre.x + 15f, centre.y - 15f, paint)
                }
            }
        }
    }

    // UI
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = { Text("ArUco Scanner") },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            )
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (!hasCameraPermission) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Camera permission required")
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Grant Permission") }
                    }
                }
                return@Scaffold
            }
            if (cameraError != null) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(cameraError!!, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { cameraError = null; startCamera() }) { Text("Retry") }
                    }
                }
                return@Scaffold
            }

            Box(Modifier.fillMaxSize().background(Color.Black)) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                StaticViewfinder()
                ArUcoOverlay(detectedMarkers)

                Column(
                    Modifier.align(Alignment.BottomCenter).padding(16.dp)
                        .background(Color.Black.copy(alpha = 0.6f), MaterialTheme.shapes.small).padding(8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    when (val state = captureState) {
                        is CaptureState.Scanning -> {
                            Text(
                                if (isTracking) "✅ ${detectedMarkers.size} marker(s) detected" else "❌ No markers",
                                color = if (isTracking) Color.Green else Color.Yellow,
                                fontSize = 16.sp, fontWeight = FontWeight.Bold
                            )
                            if (isTracking && detectedMarkers.size >= REQUIRED_MARKER_COUNT) {
                                Spacer(Modifier.height(4.dp))
                                Text("Confirming: $stableFrameCount / $CONFIRM_FRAMES", color = Color.White, fontSize = 13.sp)
                            }
                        }
                        is CaptureState.Capturing -> Text("📸 Capturing…", color = Color.Cyan, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                        is CaptureState.Captured -> {
                            Text("✅ Captured", color = Color.Green, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                            Spacer(Modifier.height(4.dp))
                            Button(onClick = { resetScan() }) { Text("Scan another") }
                        }
                        is CaptureState.Error -> {
                            Text("⚠️ ${state.message}", color = Color.Red, fontSize = 14.sp, textAlign = TextAlign.Center)
                            Spacer(Modifier.height(4.dp))
                            Button(onClick = { resetScan() }) { Text("Retry") }
                        }
                    }
                }
            }
        }
    }
}

private sealed class CaptureState {
    object Scanning : CaptureState()
    object Capturing : CaptureState()
    data class Captured(val uri: Uri) : CaptureState()
    data class Error(val message: String) : CaptureState()
}