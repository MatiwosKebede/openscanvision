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
import androidx.compose.material.icons.filled.AddCircle
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

private const val TAG = "ScannerScreen"

/* ─── Helpers ─────────────────────────────────────────────────── */

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

private fun mapBitmapPointsToSensor(
    points: List<PointF>, sensorWidth: Float, sensorHeight: Float, rotationDegrees: Int
): List<PointF> {
    if (rotationDegrees % 360 == 0) return points
    return when (rotationDegrees) {
        90 -> points.map { PointF(it.y, sensorHeight - it.x) }
        180 -> points.map { PointF(sensorWidth - it.x, sensorHeight - it.y) }
        270 -> points.map { PointF(sensorWidth - it.y, it.x) }
        else -> points
    }
}

private fun saveBitmapToGallery(context: android.content.Context, bitmap: Bitmap) {
    try {
        val filename = "OMR_${System.currentTimeMillis()}.jpg"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            uri?.let {
                val outputStream = context.contentResolver.openOutputStream(it)
                outputStream?.use { stream -> bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream) }
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.insertImage(context.contentResolver, bitmap, filename, "OMR scanned card")
        }
    } catch (e: Exception) {
        Log.e(TAG, "Error saving bitmap", e)
        Toast.makeText(context, "Failed to save image", Toast.LENGTH_SHORT).show()
    }
}

/* ─── Composable Screen ───────────────────────────────────────── */

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
    var cachedHomography by remember { mutableStateOf<Matrix?>(null) }
    var cachedCardCornersScreen by remember { mutableStateOf<List<Offset>?>(null) }
    var cachedQRCornersScreen by remember { mutableStateOf<List<Offset>?>(null) }
    var cachedMarkerCentersScreen by remember { mutableStateOf<Map<String, Offset>?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

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
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
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
                paint.style = Paint.Style.FILL; paint.alpha = 128
                canvas.drawCircle(cx, cy, 12f, paint)
                paint.style = Paint.Style.STROKE; paint.alpha = 255
            }
        }
        markerCenters?.values?.forEach { pt ->
            paint.color = Color.Magenta.toArgb(); paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.5f; paint.alpha = 64
            canvas.drawCircle(pt.x, pt.y, 20f, paint)
            paint.alpha = 230
            canvas.drawCircle(pt.x, pt.y, 8f, paint)
            val crossSize = 10f
            canvas.drawLine(pt.x - crossSize, pt.y, pt.x + crossSize, pt.y, paint)
            canvas.drawLine(pt.x, pt.y - crossSize, pt.x, pt.y + crossSize, paint)
            paint.alpha = 255; paint.style = Paint.Style.FILL
            canvas.drawCircle(pt.x, pt.y, 4f, paint)
            paint.style = Paint.Style.STROKE
        }
    }

    fun processCurrentFrame(
        sensorBitmap: Bitmap, previewSnapshot: Bitmap, homography: Matrix, qrValue: String,
        cardCornersScreen: List<Offset>?, qrCornersScreen: List<Offset>?,
        markerCentersScreen: Map<String, Offset>?
    ) {
        if (isProcessing) return
        isProcessing = true
        isScanning = true
        scanStatus = "Processing image..."

        val displayBitmap = previewSnapshot.copy(previewSnapshot.config ?: Bitmap.Config.ARGB_8888, true)
        drawScreenOverlayOnBitmap(displayBitmap, cardCornersScreen, qrCornersScreen, markerCentersScreen)

        coroutineScope.launch(Dispatchers.IO) {
            try {
                val (filledIndices, confidence, warpedBitmap) = OMRExtractor.extractCandidateMarks(
                    originalBitmap = sensorBitmap, homography = homography
                )
                withContext(Dispatchers.Main) {
                    scanResult = LocalScanResult(
                        token = qrValue,
                        filledIndices = filledIndices,
                        scanDataJson = Gson().toJson(mapOf("filled" to filledIndices, "confidence" to confidence)),
                        confidence = confidence,
                        warpedCardBitmap = warpedBitmap,
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
                    .setTargetResolution(android.util.Size(640, 480))   // lower resolution for speed
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setImageQueueDepth(1)
                    .build()

                var autoCaptureFired = false
                var frameCounter = 0

                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    frameCounter++
                    // Skip every second frame to reduce load
                    if (frameCounter % 2 != 0) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    if (isProcessing) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val mediaImage = imageProxy.image ?: run { imageProxy.close(); return@setAnalyzer }
                    val bitmap = imageProxy.toBitmap() ?: run { imageProxy.close(); return@setAnalyzer }
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val sensorWidth = imageProxy.width.toFloat()
                    val sensorHeight = imageProxy.height.toFloat()

                    val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
                    barcodeScanner.process(inputImage)
                        .addOnSuccessListener(cameraExecutor) { barcodes ->
                            val barcode = barcodes.firstOrNull()
                            val qrValue = barcode?.rawValue?.trim()
                            val qrImagePoints = barcode?.cornerPoints?.map { PointF(it.x.toFloat(), it.y.toFloat()) }

                            // ArUco detection (non‑guided, same as earlier working version)
                            val arUcoMap = CardDetector.detectArUcoMarkers(bitmap)
                            Log.d(TAG, "ArUco found: ${arUcoMap.size} markers, IDs: ${arUcoMap.keys}")

                            // Build marker screen centres
                            val markerLabels = listOf("TL", "TR", "BR", "BL")
                            val tempMarkerScreen = mutableMapOf<String, Offset>()
                            for (id in 0..3) {
                                val corners = arUcoMap[id] ?: continue
                                val cx = corners.map { it.x }.average().toFloat()
                                val cy = corners.map { it.y }.average().toFloat()
                                val centreBitmap = PointF(cx, cy)

                                val sensorPt = mapBitmapPointsToSensor(
                                    listOf(centreBitmap), sensorWidth, sensorHeight, rotationDegrees
                                ).first().let {
                                    PointF(it.x.coerceIn(0f, sensorWidth - 1f), it.y.coerceIn(0f, sensorHeight - 1f))
                                }
                                val screenOffset = mapImageToScreen(
                                    listOf(Offset(sensorPt.x, sensorPt.y)),
                                    sensorWidth, sensorHeight, previewView
                                )?.firstOrNull()
                                if (screenOffset != null) tempMarkerScreen[markerLabels[id]] = screenOffset
                            }
                            val markerScreenMap = tempMarkerScreen.ifEmpty { null }
                            Log.d(TAG, "markerScreenMap: $markerScreenMap")

                            // Homography only if all four markers
                            val homography = if (arUcoMap.size == 4) {
                                CardDetector.buildHomographyFromArUco(arUcoMap)
                            } else null

                            // Card corners from homography, or bounding box of ArUco corners
                            var imageCorners: List<PointF>? = null
                            if (homography != null) {
                                val templateCardCorners = listOf(
                                    PointF(0f, 0f),
                                    PointF(Templates.REF_WIDTH.toFloat(), 0f),
                                    PointF(Templates.REF_WIDTH.toFloat(), Templates.REF_HEIGHT.toFloat()),
                                    PointF(0f, Templates.REF_HEIGHT.toFloat())
                                )
                                imageCorners = CardDetector.predictImagePoints(templateCardCorners, homography)
                            } else if (arUcoMap.size == 4) {
                                val allPts = arUcoMap.values.flatten()
                                val minX = allPts.minOf { it.x }
                                val minY = allPts.minOf { it.y }
                                val maxX = allPts.maxOf { it.x }
                                val maxY = allPts.maxOf { it.y }
                                imageCorners = listOf(
                                    PointF(minX, minY), PointF(maxX, minY),
                                    PointF(maxX, maxY), PointF(minX, maxY)
                                )
                            }

                            var finalCardCorners: List<Offset>? = null
                            if (imageCorners != null && imageCorners.size == 4) {
                                val sensorCorners = mapBitmapPointsToSensor(
                                    imageCorners, sensorWidth, sensorHeight, rotationDegrees
                                ).map { PointF(it.x.coerceIn(0f, sensorWidth - 1f), it.y.coerceIn(0f, sensorHeight - 1f)) }
                                finalCardCorners = mapImageToScreen(
                                    sensorCorners.map { Offset(it.x, it.y) },
                                    sensorWidth, sensorHeight, previewView
                                )
                            }
                            Log.d(TAG, "finalCardCorners: $finalCardCorners")

                            // QR screen corners
                            val qrScreenCorners = if (qrValue != null && qrImagePoints != null && qrImagePoints.size == 4) {
                                mapImageToScreen(
                                    qrImagePoints.map {
                                        Offset(
                                            it.x.coerceIn(0f, sensorWidth - 1f),
                                            it.y.coerceIn(0f, sensorHeight - 1f)
                                        )
                                    },
                                    sensorWidth, sensorHeight, previewView
                                )
                            } else null
                            Log.d(TAG, "qrScreenCorners: $qrScreenCorners")

                            // Update live preview state
                            coroutineScope.launch(Dispatchers.Main) {
                                rawCardCorners = finalCardCorners
                                rawQRCorners = qrScreenCorners
                                rawMarkerCenters = markerScreenMap
                                qrDetected = qrValue != null
                                isTracking = finalCardCorners != null || arUcoMap.size == 4

                                latestToken = qrValue
                                cachedHomography = homography
                                cachedCardCornersScreen = finalCardCorners
                                cachedQRCornersScreen = qrScreenCorners
                                cachedMarkerCentersScreen = markerScreenMap
                                currentFrameBitmap?.recycle()
                                currentFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)

                                // Reset auto‑capture when markers are lost (so it can fire again later)
                                if (arUcoMap.size < 4) autoCaptureFired = false
                            }

                            // Auto‑capture (once per card presentation, when all markers + QR are present)
                            if (homography != null && qrValue != null && !isProcessing && !autoCaptureFired) {
                                autoCaptureFired = true
                                val sensorFrameCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
                                coroutineScope.launch(Dispatchers.Main) {
                                    val previewSnapshot = previewView.bitmap
                                    if (previewSnapshot != null) {
                                        processCurrentFrame(
                                            sensorBitmap = sensorFrameCopy,
                                            previewSnapshot = previewSnapshot,
                                            homography = homography,
                                            qrValue = qrValue,
                                            cardCornersScreen = finalCardCorners,
                                            qrCornersScreen = qrScreenCorners,
                                            markerCentersScreen = markerScreenMap
                                        )
                                    } else {
                                        sensorFrameCopy.recycle()
                                        scanStatus = "Preview not ready, try again"
                                    }
                                }
                            }

                            imageProxy.close()
                        }
                        .addOnFailureListener(cameraExecutor) { e ->
                            Log.e(TAG, "QR detection failed", e)
                            // Edge‑detection fallback
                            val downsampled = downsampleBitmap(bitmap, 800, 450)
                            val cardCorners = CardDetector.detectCardCorners(downsampled)
                            if (cardCorners != null && cardCorners.size == 4) {
                                val scaleX = bitmap.width.toFloat() / downsampled.width.toFloat()
                                val scaleY = bitmap.height.toFloat() / downsampled.height.toFloat()
                                val scaledCorners = cardCorners.map { PointF(it.x * scaleX, it.y * scaleY) }
                                val sensorSpace = mapBitmapPointsToSensor(
                                    scaledCorners, sensorWidth, sensorHeight, rotationDegrees
                                )
                                val screenCorners = mapImageToScreen(
                                    sensorSpace.map { Offset(it.x, it.y) },
                                    sensorWidth, sensorHeight, previewView
                                )
                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = screenCorners
                                    isTracking = true
                                    rawQRCorners = null
                                    qrDetected = false
                                    rawMarkerCenters = null
                                    autoCaptureFired = false
                                }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) {
                                    isTracking = false
                                    rawCardCorners = null
                                    rawQRCorners = null
                                    rawMarkerCenters = null
                                    qrDetected = false
                                    autoCaptureFired = false
                                }
                            }
                            downsampled.recycle()
                            imageProxy.close()
                        }
                }

                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
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

    // Full‑screen viewer
    fullScreenBitmap?.let { bmp ->
        var scale by remember { mutableFloatStateOf(1f) }
        var offset by remember { mutableStateOf(Offset.Zero) }
        Dialog(
            onDismissRequest = { fullScreenBitmap = null },
            properties = DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Full screen",
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer {
                            scaleX = scale; scaleY = scale
                            translationX = offset.x; translationY = offset.y
                        }
                        .pointerInput(Unit) {
                            detectTransformGestures { _, pan, zoom, _ ->
                                scale = (scale * zoom).coerceIn(1f, 5f)
                                offset = Offset(offset.x + pan.x, offset.y + pan.y)
                            }
                        }
                )
                IconButton(
                    onClick = { saveBitmapToGallery(context, bmp); Toast.makeText(context, "Saved", Toast.LENGTH_SHORT).show() },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(16.dp)
                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                ) { Icon(Icons.Filled.AddCircle, "Save", tint = Color.White) }
                Box(modifier = Modifier.fillMaxSize().clickable { fullScreenBitmap = null })
            }
        }
    }

    Scaffold(containerColor = MaterialTheme.colorScheme.background) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (!hasCameraPermission) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Camera permission required", color = Color.White)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Grant Permission") }
                    }
                }
                return@Scaffold
            }
            if (cameraError != null) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(cameraError!!, color = Color.White, textAlign = TextAlign.Center)
                        Spacer(Modifier.height(8.dp))
                        Button(onClick = { cameraError = null; startCamera() }) { Text("Retry") }
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
                    bubblePositions = null, bubbleStatus = null,
                    isTracking = isTracking, qrDetected = qrDetected,
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
                statusText,
                color = if (isTracking && qrDetected) Color.Green else Color.Yellow,
                fontSize = 14.sp, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
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
                                value = manualSerial, onValueChange = { manualSerial = it },
                                label = { Text("Enter Serial Code") }, modifier = Modifier.fillMaxWidth(), singleLine = true
                            )
                            Spacer(Modifier.height(8.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                Button(onClick = {
                                    if (manualSerial.isNotBlank()) {
                                        scanResult = LocalScanResult(manualSerial, emptyList(), "{}", 1f)
                                        showManualEntry = false
                                    }
                                }, modifier = Modifier.weight(1f), enabled = manualSerial.isNotBlank()) { Text("Apply") }
                                Button(onClick = { showManualEntry = false }, modifier = Modifier.weight(1f)) { Text("Cancel") }
                            }
                        }
                    }
                }
                Spacer(Modifier.height(16.dp))
                Button(
                    onClick = {
                        val homography = cachedHomography
                        val sensorFrame = currentFrameBitmap
                        val previewSnapshot = previewView.bitmap
                        when {
                            homography == null || latestToken == null || sensorFrame == null ->
                                scanStatus = "No frame or QR code detected yet"
                            previewSnapshot == null -> scanStatus = "Preview not ready"
                            else -> processCurrentFrame(
                                sensorBitmap = sensorFrame, previewSnapshot = previewSnapshot,
                                homography = homography, qrValue = latestToken!!,
                                cardCornersScreen = cachedCardCornersScreen,
                                qrCornersScreen = cachedQRCornersScreen,
                                markerCentersScreen = cachedMarkerCentersScreen
                            )
                        }
                    },
                    modifier = Modifier.fillMaxWidth(0.7f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFEB914)),
                    enabled = cachedHomography != null && latestToken != null && currentFrameBitmap != null
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
                                Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Processed", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    if (result.warpedCardBitmap != null) {
                                        Image(bitmap = result.warpedCardBitmap.asImageBitmap(), contentDescription = "Processed",
                                            modifier = Modifier.fillMaxWidth().height(180.dp).clip(RoundedCornerShape(8.dp))
                                                .clickable { fullScreenBitmap = result.warpedCardBitmap })
                                    } else {
                                        Box(modifier = Modifier.fillMaxWidth().height(180.dp).background(Color.Gray), contentAlignment = Alignment.Center) {
                                            Text("No image", color = Color.White, fontSize = 12.sp)
                                        }
                                    }
                                }
                                Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("Original", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    if (result.originalBitmap != null) {
                                        Image(bitmap = result.originalBitmap.asImageBitmap(), contentDescription = "Original",
                                            modifier = Modifier.fillMaxWidth().height(180.dp).clip(RoundedCornerShape(8.dp))
                                                .clickable { fullScreenBitmap = result.originalBitmap })
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
                                Text(result.scanDataJson, fontFamily = FontFamily.Monospace, fontSize = 11.sp,
                                    modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)).padding(8.dp))
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