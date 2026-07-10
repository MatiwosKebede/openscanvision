package org.openscanvision.ui.screens

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountBox
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
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
import java.io.OutputStream
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

private const val TAG = "ScannerScreen"
private const val MOTION_EPSILON_PX = 15f
private const val STABLE_FRAMES_REQUIRED = 4

/** Sorts four points into TL, TR, BR, BL order (used only if edge detection fallback is ever needed). */
private fun sortCornersTopLeftFirst(corners: List<PointF>): List<PointF> {
    if (corners.size != 4) return corners
    val sortedBySum = corners.sortedBy { it.x + it.y }
    val tl = sortedBySum.first()
    val br = sortedBySum.last()
    val remaining = corners.filter { it != tl && it != br }
    val sortedByDiff = remaining.sortedBy { it.y - it.x }
    val tr = sortedByDiff.first()
    val bl = sortedByDiff.last()
    return listOf(tl, tr, br, bl)
}

/**
 * Maps points from the original sensor image space to the rotated bitmap space,
 * matching the transformation done by `imageProxy.toBitmap()`. Used ONLY to bring
 * ML-Kit's QR corner points into the same coordinate space as `bitmap` for the OMR
 * homography/processing pipeline (confirmed correct — do not change this half).
 */
private fun mapSensorPointsToBitmap(
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
 * The exact inverse of [mapSensorPointsToBitmap]: brings points computed in rotated
 * bitmap space (e.g. card corners / markers derived from the homography, which
 * operates on `bitmap`) back into the ORIGINAL sensor coordinate space.
 *
 * This is needed specifically for the live preview overlay: `mapImageToScreen` (in
 * ui.utils) is a CameraX-style coordinate mapper built around the raw, pre-rotation
 * ImageProxy dimensions (the same convention CameraX's own OutputTransform APIs use),
 * and does its own rotation-aware mapping into PreviewView space internally. Feeding
 * it already-rotated points plus swapped (rotated) width/height double-applies the
 * rotation, which is what was producing a scattered/wrong overlay even though the
 * OMR processing pipeline (which never touches mapImageToScreen) was fine.
 */
private fun mapBitmapPointsToSensor(
    points: List<PointF>,
    sensorWidth: Float,
    sensorHeight: Float,
    rotationDegrees: Int
): List<PointF> {
    if (rotationDegrees % 360 == 0) return points
    return when (rotationDegrees) {
        90 -> points.map { PointF(it.y, sensorHeight - it.x) }
        180 -> points.map { PointF(sensorWidth - it.x, sensorHeight - it.y) }
        270 -> points.map { PointF(sensorWidth - it.y, it.x) }
        else -> points
    }
}

/** Saves a Bitmap to the device gallery. */
private fun saveBitmapToGallery(context: android.content.Context, bitmap: Bitmap) {
    try {
        val filename = "OMR_${System.currentTimeMillis()}.jpg"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }
            val resolver = context.contentResolver
            val uri: Uri? = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            uri?.let {
                val outputStream: OutputStream? = resolver.openOutputStream(it)
                outputStream?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                }
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.insertImage(
                context.contentResolver,
                bitmap,
                filename,
                "OMR scanned card"
            )
        }
    } catch (e: Exception) {
        Log.e(TAG, "Error saving bitmap", e)
        Toast.makeText(context, "Failed to save image", Toast.LENGTH_SHORT).show()
    }
}

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
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasCameraPermission = it }

    var isScanning by remember { mutableStateOf(false) }
    var manualSerial by remember { mutableStateOf("") }
    var showManualEntry by remember { mutableStateOf(false) }
    var scanStatus by remember { mutableStateOf("") }
    var scanResult by remember { mutableStateOf<LocalScanResult?>(null) }

    var rawCardCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawQRCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawMarkerCenters by remember { mutableStateOf<Map<String, Offset>?>(null) }
    var qrDetected by remember { mutableStateOf(false) }
    var isTracking by remember { mutableStateOf(false) }

    var latestToken by remember { mutableStateOf<String?>(null) }
    var currentFrameBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var currentQrPoints by remember { mutableStateOf<List<PointF>?>(null) }
    var cachedCardImageCorners by remember { mutableStateOf<List<PointF>?>(null) }
    var cachedCardCornersScreen by remember { mutableStateOf<List<Offset>?>(null) }
    var cachedQRCornersScreen by remember { mutableStateOf<List<Offset>?>(null) }
    var cachedMarkerCentersScreen by remember { mutableStateOf<Map<String, Offset>?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    var currentCardImageCorners by remember { mutableStateOf<List<PointF>?>(null) }

    // Full‑screen image viewer state
    var fullScreenBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val smoothCorners = rememberSmoothCorners(rawCardCorners, smoothing = 0.3f)
    val smoothQRCorners = rememberSmoothCorners(rawQRCorners, smoothing = 0.3f)
    val rawMarkerList = rawMarkerCenters?.values?.toList()
    val smoothMarkerList = rememberSmoothCorners(rawMarkerList, smoothing = 0.3f)
    val smoothMarkerMap = remember(rawMarkerCenters, smoothMarkerList) {
        if (rawMarkerCenters != null && smoothMarkerList != null && rawMarkerCenters!!.size == smoothMarkerList.size) {
            rawMarkerCenters!!.keys.zip(smoothMarkerList).toMap()
        } else null
    }

    val barcodeScanner = remember { BarcodeScanning.getClient() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            barcodeScanner.close()
            currentFrameBitmap?.recycle()
            fullScreenBitmap?.recycle()
        }
    }

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

                val cx = qrPts.map { it.x }.average().toFloat()
                val cy = qrPts.map { it.y }.average().toFloat()
                paint.style = Paint.Style.FILL
                paint.alpha = 128
                canvas.drawCircle(cx, cy, 12f, paint)
                paint.style = Paint.Style.STROKE
                paint.alpha = 255
            }
        }

        markerCenters?.values?.forEach { pt ->
            paint.color = Color.Magenta.toArgb()
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.5f
            paint.alpha = 64
            canvas.drawCircle(pt.x, pt.y, 20f, paint)
            paint.alpha = 230
            canvas.drawCircle(pt.x, pt.y, 8f, paint)
            val crossSize = 10f
            canvas.drawLine(pt.x - crossSize, pt.y, pt.x + crossSize, pt.y, paint)
            canvas.drawLine(pt.x, pt.y - crossSize, pt.x, pt.y + crossSize, paint)
            paint.alpha = 255
            paint.style = Paint.Style.FILL
            canvas.drawCircle(pt.x, pt.y, 4f, paint)
            paint.style = Paint.Style.STROKE
        }
    }

    fun processCurrentFrame(
        sensorBitmap: Bitmap,
        previewSnapshot: Bitmap,
        qrValue: String,
        qrPoints: List<PointF>?,
        cardCornersImage: List<PointF>?,
        cardCornersScreen: List<Offset>?,
        qrCornersScreen: List<Offset>?,
        markerCentersScreen: Map<String, Offset>?
    ) {
        if (isProcessing) return
        isProcessing = true
        isScanning = true
        scanStatus = "Processing image..."

        val displayBitmap = previewSnapshot.copy(
            previewSnapshot.config ?: Bitmap.Config.ARGB_8888, true
        )
        drawScreenOverlayOnBitmap(displayBitmap, cardCornersScreen, qrCornersScreen, markerCentersScreen)

        coroutineScope.launch(Dispatchers.IO) {
            try {
                val (filledIndices, confidence, warpedBitmap) = OMRExtractor.extractCandidateMarks(
                    originalBitmap = sensorBitmap,
                    cardCornersImage = cardCornersImage,
                    qrCorners = qrPoints ?: emptyList()
                )

                // Rotate 180° to fix upside‑down processed image
                val finalWarped = if (warpedBitmap != null) {
                    val matrix = Matrix()
                    matrix.postRotate(180f)
                    Bitmap.createBitmap(warpedBitmap, 0, 0, warpedBitmap.width, warpedBitmap.height, matrix, true)
                        .also { warpedBitmap.recycle() }
                } else null

                withContext(Dispatchers.Main) {
                    scanResult = LocalScanResult(
                        token = qrValue,
                        filledIndices = filledIndices,
                        scanDataJson = Gson().toJson(mapOf(
                            "filled" to filledIndices,
                            "confidence" to confidence
                        )),
                        confidence = confidence,
                        warpedCardBitmap = finalWarped,
                        originalBitmap = displayBitmap
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
                    val imageWidth = bitmap.width.toFloat()
                    val imageHeight = bitmap.height.toFloat()
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val sensorWidth = imageProxy.width.toFloat()
                    val sensorHeight = imageProxy.height.toFloat()

                    val downsampled = downsampleBitmap(bitmap, 800, 450)
                    val processWidth = downsampled.width.toFloat()
                    val processHeight = downsampled.height.toFloat()

                    val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
                    barcodeScanner.process(inputImage)
                        .addOnSuccessListener(cameraExecutor) { barcodes ->
                            val barcode = barcodes.firstOrNull()
                            val qrValue = barcode?.rawValue?.trim()
                            val qrImagePoints = barcode?.cornerPoints?.map { PointF(it.x.toFloat(), it.y.toFloat()) }

                            val rotatedQrPoints = if (qrImagePoints != null) {
                                mapSensorPointsToBitmap(qrImagePoints, sensorWidth, sensorHeight, rotationDegrees)
                            } else null

                            var finalCardCorners: List<Offset>? = null
                            var markerScreenMap: Map<String, Offset>? = null
                            var template: CardTemplate? = null
                            var imageCorners: List<PointF>? = null
                            var warpImageCorners: List<PointF>? = null

                            if (qrValue != null && rotatedQrPoints != null && rotatedQrPoints.size == 4) {
                                template = Templates.fromPrefix(qrValue)

                                if (template != null) {
                                    val homography = OMRExtractor.computeMarkerHomography(bitmap, rotatedQrPoints, template)
                                    if (homography != null) {
                                        val templateCorners = listOf(
                                            PointF(0f, 0f),
                                            PointF(Templates.REF_WIDTH.toFloat(), 0f),
                                            PointF(Templates.REF_WIDTH.toFloat(), Templates.REF_HEIGHT.toFloat()),
                                            PointF(0f, Templates.REF_HEIGHT.toFloat())
                                        )
                                        imageCorners = CardDetector.predictImagePoints(templateCorners, homography)
                                        warpImageCorners = imageCorners

                                        val sensorCardCorners = mapBitmapPointsToSensor(
                                            imageCorners!!, sensorWidth, sensorHeight, rotationDegrees
                                        )
                                        val screenCorners = mapImageToScreen(
                                            sensorCardCorners.map { Offset(it.x, it.y) },
                                            sensorWidth, sensorHeight, previewView
                                        )
                                        if (screenCorners != null && screenCorners.size == 4) {
                                            finalCardCorners = screenCorners
                                        }

                                        val markerRefs = template.markerRefPositions ?: Templates.SHARED_MARKER_CORNERS
                                        val predicted = CardDetector.predictImagePoints(markerRefs, homography)
                                        val detected = CardDetector.detectRefMarkersNearPredicted(bitmap, predicted)
                                        val markerLabels = listOf("TL", "TR", "BR", "BL")
                                        val tempMarkerScreen = mutableMapOf<String, Offset>()
                                        for (i in markerLabels.indices) {
                                            val pt = detected.getOrNull(i) ?: continue
                                            val sensorPt = mapBitmapPointsToSensor(
                                                listOf(pt), sensorWidth, sensorHeight, rotationDegrees
                                            ).first()
                                            val screenOffset = mapImageToScreen(
                                                listOf(Offset(sensorPt.x, sensorPt.y)),
                                                sensorWidth, sensorHeight, previewView
                                            )?.firstOrNull()
                                            if (screenOffset != null) tempMarkerScreen[markerLabels[i]] = screenOffset
                                        }
                                        markerScreenMap = tempMarkerScreen.ifEmpty { null }
                                    }
                                }
                            }

                            if (finalCardCorners == null) {
                                val cardCorners = CardDetector.detectCardCorners(downsampled)
                                if (cardCorners != null && cardCorners.size == 4) {
                                    val scaleX = imageWidth / processWidth
                                    val scaleY = imageHeight / processHeight
                                    val bitmapSpaceCorners = cardCorners.map { PointF(it.x * scaleX, it.y * scaleY) }

                                    if (warpImageCorners == null) {
                                        warpImageCorners = sortCornersTopLeftFirst(bitmapSpaceCorners)
                                    }

                                    val sensorSpaceCorners = mapBitmapPointsToSensor(
                                        bitmapSpaceCorners, sensorWidth, sensorHeight, rotationDegrees
                                    )
                                    finalCardCorners = sensorSpaceCorners.map { Offset(it.x, it.y) }.let {
                                        mapImageToScreen(it, sensorWidth, sensorHeight, previewView)
                                    }
                                }
                            }

                            val qrScreenCorners = if (qrValue != null && qrImagePoints != null && qrImagePoints.size == 4) {
                                mapImageToScreen(
                                    qrImagePoints.map { Offset(it.x, it.y) },
                                    sensorWidth, sensorHeight, previewView
                                )
                            } else null

                            coroutineScope.launch(Dispatchers.Main) {
                                cachedCardImageCorners = warpImageCorners
                                currentCardImageCorners = warpImageCorners

                                latestToken = qrValue
                                currentQrPoints = rotatedQrPoints
                                cachedCardCornersScreen = finalCardCorners
                                cachedQRCornersScreen = qrScreenCorners
                                cachedMarkerCentersScreen = markerScreenMap
                                currentFrameBitmap?.recycle()
                                currentFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)

                                rawCardCorners = finalCardCorners
                                rawQRCorners = qrScreenCorners
                                rawMarkerCenters = markerScreenMap
                                qrDetected = qrValue != null
                                isTracking = finalCardCorners != null
                            }

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
                                    val capturedQRCorners = qrScreenCorners
                                    val capturedMarkerMap = markerScreenMap
                                    val capturedWarpCorners = warpImageCorners
                                    coroutineScope.launch(Dispatchers.Main) {
                                        val previewSnapshot = previewView.bitmap
                                        if (previewSnapshot != null) {
                                            processCurrentFrame(
                                                sensorBitmap = sensorFrameCopy,
                                                previewSnapshot = previewSnapshot,
                                                qrValue = qrValue,
                                                qrPoints = rotatedQrPoints,
                                                cardCornersImage = capturedWarpCorners,
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
                            val cardCorners = CardDetector.detectCardCorners(downsampled)
                            if (cardCorners != null && cardCorners.size == 4) {
                                val scaleX = imageWidth / processWidth
                                val scaleY = imageHeight / processHeight
                                val bitmapSpaceCorners = cardCorners.map { PointF(it.x * scaleX, it.y * scaleY) }
                                val sorted = sortCornersTopLeftFirst(bitmapSpaceCorners)

                                val sensorSpaceCorners = mapBitmapPointsToSensor(
                                    bitmapSpaceCorners, sensorWidth, sensorHeight, rotationDegrees
                                )
                                val screenCorners = mapImageToScreen(
                                    sensorSpaceCorners.map { Offset(it.x, it.y) },
                                    sensorWidth, sensorHeight, previewView
                                )

                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = screenCorners
                                    isTracking = true
                                    rawQRCorners = null
                                    qrDetected = false
                                    rawMarkerCenters = null
                                    stableFrameCount = 0
                                    lastStableCorners = null
                                    autoCaptureTriggered = false
                                    currentCardImageCorners = sorted
                                    cachedCardImageCorners = sorted
                                    cachedCardCornersScreen = screenCorners
                                    cachedQRCornersScreen = null
                                    cachedMarkerCentersScreen = null
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
                                    currentCardImageCorners = null
                                    cachedCardImageCorners = null
                                    cachedCardCornersScreen = null
                                    cachedQRCornersScreen = null
                                    cachedMarkerCentersScreen = null
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

    // ─── Full‑screen viewer with zoom & save ────────────────────────────
    fullScreenBitmap?.let { bmp ->
        var scale by remember { mutableFloatStateOf(1f) }
        var offset by remember { mutableStateOf(Offset.Zero) }

        Dialog(
            onDismissRequest = { fullScreenBitmap = null },
            properties = DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
            ) {
                // Zoomable & pannable image
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Full screen",
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer {
                            scaleX = scale
                            scaleY = scale
                            translationX = offset.x
                            translationY = offset.y
                        }
                        .pointerInput(Unit) {
                            detectTransformGestures { _, pan, zoom, _ ->
                                scale = (scale * zoom).coerceIn(1f, 5f)
                                offset = Offset(
                                    x = offset.x + pan.x,
                                    y = offset.y + pan.y
                                )
                            }
                        }
                )

                // Save button (top‑right corner)
                IconButton(
                    onClick = {
                        saveBitmapToGallery(context, bmp)
                        Toast.makeText(context, "Image saved to gallery", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(16.dp)
                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                ) {
                    Icon(
                        Icons.Default.AccountBox,
                        contentDescription = "Save",
                        tint = Color.White
                    )
                }

                // Tap to close (outside the image)
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clickable { fullScreenBitmap = null }
                )
            }
        }
    }

    // ─── UI ──────────────────────────────────────────────────────────────
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
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
                                ) { Text("Apply") }
                                Button(onClick = { showManualEntry = false }, modifier = Modifier.weight(1f)) { Text("Cancel") }
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
                        val imageCorners = cachedCardImageCorners
                        val cardCornersScreen = cachedCardCornersScreen
                        val qrCornersScreen = cachedQRCornersScreen
                        val markerCentersScreen = cachedMarkerCentersScreen
                        when {
                            token == null || sensorFrame == null -> scanStatus = "No frame or QR code detected yet"
                            previewSnapshot == null -> scanStatus = "Preview not ready, try again"
                            else -> {
                                processCurrentFrame(
                                    sensorBitmap = sensorFrame,
                                    previewSnapshot = previewSnapshot,
                                    qrValue = token,
                                    qrPoints = currentQrPoints,
                                    cardCornersImage = imageCorners,
                                    cardCornersScreen = cardCornersScreen,
                                    qrCornersScreen = qrCornersScreen,
                                    markerCentersScreen = markerCentersScreen
                                )
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(0.7f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFEB914)),
                    enabled = latestToken != null && currentFrameBitmap != null
                ) { Text("Capture & Verify", color = Color.Black, fontWeight = FontWeight.Bold) }
            }

            scanResult?.let { result ->
                AlertDialog(
                    onDismissRequest = {
                        scanResult = null; isScanning = false
                        result.warpedCardBitmap?.recycle(); result.originalBitmap?.recycle()
                    },
                    text = {
                        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                // Processed image – clickable for full screen
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
                                                .clickable { fullScreenBitmap = result.warpedCardBitmap }
                                        )
                                    } else {
                                        Box(modifier = Modifier.fillMaxWidth().height(180.dp).background(Color.Gray), contentAlignment = Alignment.Center) {
                                            Text("No image", color = Color.White, fontSize = 12.sp)
                                        }
                                    }
                                }
                                // Original image – also clickable for full screen
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
                                                .clickable { fullScreenBitmap = result.originalBitmap }
                                        )
                                    } else {
                                        Box(modifier = Modifier.fillMaxWidth().height(180.dp).background(Color.Gray), contentAlignment = Alignment.Center) {
                                            Text("No image", color = Color.White, fontSize = 12.sp)
                                        }
                                    }
                                }
                            }
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
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
                            TextButton(onClick = { showDetails = !showDetails }) { Text(if (showDetails) "Hide raw data" else "Show raw data") }
                            if (showDetails) {
                                Text(
                                    result.scanDataJson,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp,
                                    modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)).padding(8.dp)
                                )
                            }
                        }
                    },
                    confirmButton = {
                        Button(onClick = {
                            scanResult = null; isScanning = false
                            result.warpedCardBitmap?.recycle(); result.originalBitmap?.recycle()
                        }) { Text("OK") }
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