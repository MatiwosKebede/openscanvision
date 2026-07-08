// ScannerScreen.kt
package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
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
import org.openscanvision.omr.*
import org.openscanvision.ui.components.*
import org.openscanvision.ui.utils.*
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

private const val TAG = "ScannerScreen"
private const val MOTION_EPSILON_PX = 6f
private const val STABLE_FRAMES_REQUIRED = 4

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

    // ─── Live tracking states ──────────────────────────────────────
    var rawCardCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawQRCorners by remember { mutableStateOf<List<Offset>?>(null) }
    var rawMarkerCenters by remember { mutableStateOf<Map<String, Offset>?>(null) }
    var qrDetected by remember { mutableStateOf(false) }
    var rawBubblePositions by remember { mutableStateOf<List<Offset>?>(null) }
    var rawBubbleStatus by remember { mutableStateOf<List<Boolean>?>(null) }
    var isTracking by remember { mutableStateOf(false) }

    var latestToken by remember { mutableStateOf<String?>(null) }
    var latestBubbleGroups by remember { mutableStateOf<List<IntRange>?>(null) }
    var latestConfidence by remember { mutableStateOf(1f) }

    var isProcessing by remember { mutableStateOf(false) }
    val frameQueue = remember { mutableListOf<Bitmap>() }
    val MAX_QUEUE = 5

    val previewView = remember { PreviewView(context) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    // ─── Smoothing ──────────────────────────────────────────────────
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
    val smoothBubbles = rememberSmoothCorners(rawBubblePositions, smoothing = 0.3f)

    // ─── Camera executor and scanner ──────────────────────────────
    val barcodeScanner = remember { BarcodeScanning.getClient() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            barcodeScanner.close()
        }
    }

    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()

                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

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

                    val downsampled = downsampleBitmap(bitmap, 800, 450)
                    val processWidth = downsampled.width.toFloat()
                    val processHeight = downsampled.height.toFloat()

                    val cardCorners = CardDetector.detectCardCorners(downsampled)

                    val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                    barcodeScanner.process(inputImage)
                        .addOnSuccessListener(cameraExecutor) { barcodes ->
                            val barcode = barcodes.firstOrNull()
                            val qrValue = barcode?.rawValue?.trim()
                            val qrImagePoints = barcode?.cornerPoints?.map { PointF(it.x.toFloat(), it.y.toFloat()) }

                            var finalCardCorners: List<Offset>? = null
                            var finalQRCorners: List<Offset>? = null
                            var finalBubbles: List<PointF>? = null
                            var template: CardTemplate? = null
                            var markerScreenMap: Map<String, Offset>? = null
                            var groups: List<IntRange>? = null
                            var statuses: List<Boolean>? = null

                            if (cardCorners != null && cardCorners.size == 4) {
                                val scaleX = imageWidth / processWidth
                                val scaleY = imageHeight / processHeight
                                finalCardCorners = cardCorners.map { Offset(it.x * scaleX, it.y * scaleY) }
                            }

                            if (qrValue != null && qrImagePoints != null && qrImagePoints.size == 4) {
                                finalQRCorners = qrImagePoints.map { Offset(it.x, it.y) }
                                template = Templates.fromPrefix(qrValue)

                                if (template != null) {
                                    val qrOnlyHomography = CardDetector.buildRefinedHomography(
                                        template.qrRefCorners, qrImagePoints
                                    )
                                    if (qrOnlyHomography != null) {
                                        val templatePts = template.qrRefCorners.toMutableList()
                                        val imagePts = qrImagePoints.toMutableList()

                                        val markerRefs = template.markerRefPositions
                                        if (!markerRefs.isNullOrEmpty()) {
                                            val predicted = CardDetector.predictImagePoints(markerRefs, qrOnlyHomography)
                                            val detected = CardDetector.detectRefMarkersNearPredicted(bitmap, predicted)
                                            val markerLabels = listOf("TL", "BR", "BL")
                                            val detectedMap = detected.mapIndexedNotNull { i, pt ->
                                                if (pt != null) markerLabels[i] to Offset(pt.x, pt.y) else null
                                            }.toMap()
                                            if (detectedMap.isNotEmpty()) {
                                                val markerImagePoints = detectedMap.values.toList()
                                                val markerScreenPoints = mapImageToScreen(markerImagePoints, imageWidth, imageHeight, previewView)
                                                if (markerScreenPoints != null) {
                                                    markerScreenMap = detectedMap.keys.zip(markerScreenPoints).toMap()
                                                }
                                            }
                                            detected.forEachIndexed { i, pt ->
                                                if (pt != null) {
                                                    templatePts.add(markerRefs[i])
                                                    imagePts.add(pt)
                                                }
                                            }
                                        }

                                        val refined = CardDetector.buildRefinedHomography(templatePts, imagePts)
                                            ?: qrOnlyHomography

                                        val srcFloats = FloatArray(template.bubblePositions.size * 2)
                                        template.bubblePositions.forEachIndexed { i, p ->
                                            srcFloats[i * 2] = p.x
                                            srcFloats[i * 2 + 1] = p.y
                                        }
                                        val dstFloats = srcFloats.copyOf()
                                        refined.mapPoints(dstFloats)
                                        finalBubbles = template.bubblePositions.indices.map {
                                            PointF(dstFloats[it * 2], dstFloats[it * 2 + 1])
                                        }
                                    }
                                }
                            }

                            // Persistence
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

                            if (finalQRCorners != null) {
                                val screenQR = mapImageToScreen(finalQRCorners, imageWidth, imageHeight, previewView)
                                coroutineScope.launch(Dispatchers.Main) { rawQRCorners = screenQR }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) { rawQRCorners = null }
                            }

                            var groupsForConfidence: List<IntRange>? = null

                            if (finalCardCorners != null) {
                                val screenCorners = mapImageToScreen(finalCardCorners, imageWidth, imageHeight, previewView)
                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = screenCorners
                                    isTracking = true
                                }

                                if (finalBubbles != null && template != null) {
                                    val darkness = BubbleAnalyzer.sampleDarkness(bitmap, finalBubbles!!, radiusPx = 15)
                                    val groupsTmp = template!!.bubbleGroups
                                        ?: listOf(finalBubbles!!.indices.first..finalBubbles!!.indices.last)
                                    groupsForConfidence = groupsTmp
                                    statuses = BubbleAnalyzer.classifyByGroup(darkness, groupsTmp).toList()
                                    groups = groupsTmp

                                    val screenBubbles = mapImageToScreen(
                                        finalBubbles!!.map { Offset(it.x, it.y) }, imageWidth, imageHeight, previewView
                                    )
                                    coroutineScope.launch(Dispatchers.Main) {
                                        rawBubblePositions = screenBubbles
                                        rawBubbleStatus = statuses
                                        latestToken = qrValue
                                        latestBubbleGroups = groupsTmp
                                        latestConfidence = BubbleAnalyzer.computeConfidence(statuses!!.toBooleanArray(), groupsTmp)
                                    }
                                } else {
                                    coroutineScope.launch(Dispatchers.Main) {
                                        rawBubblePositions = null
                                        rawBubbleStatus = null
                                    }
                                }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) {
                                    isTracking = false
                                    rawCardCorners = null
                                    rawBubblePositions = null
                                    rawBubbleStatus = null
                                }
                            }

                            // ─── Auto‑capture ──────────────────────
                            if (finalCardCorners != null && finalBubbles != null && template != null && !autoCaptureTriggered) {
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
                                    val capturedToken = qrValue
                                    val capturedStatuses = statuses
                                    val capturedGroups = groupsForConfidence
                                    coroutineScope.launch(Dispatchers.Main) {
                                        isScanning = true
                                        scanStatus = "Auto‑capturing..."
                                        isProcessing = true
                                        delay(200)
                                        val averaged = if (frameQueue.size >= 3) BubbleAnalyzer.averageBitmaps(frameQueue) else frameQueue.lastOrNull()
                                        if (averaged == null || capturedToken == null || capturedStatuses == null || capturedGroups == null) {
                                            isScanning = false
                                            isProcessing = false
                                            scanStatus = "No frames available"
                                            autoCaptureTriggered = false
                                            return@launch
                                        }
                                        val (filledIndices, confidence) = OMRExtractor.extractMarksWithConfidence(
                                            averaged,
                                            qrImagePoints ?: return@launch,
                                            template
                                        )
                                        scanResult = LocalScanResult(
                                            token = capturedToken,
                                            filledIndices = filledIndices,
                                            scanDataJson = Gson().toJson(mapOf("filledIndices" to filledIndices)),
                                            confidence = confidence
                                        )
                                        isScanning = false
                                        isProcessing = false
                                        scanStatus = "✅ Auto‑captured"
                                        frameQueue.clear()
                                        autoCaptureTriggered = false
                                    }
                                }
                            } else {
                                stableFrameCount = 0
                                lastStableCorners = null
                                autoCaptureTriggered = false
                            }

                            frameQueue.add(bitmap)
                            if (frameQueue.size > MAX_QUEUE) frameQueue.removeAt(0)
                            downsampled.recycle()
                            imageProxy.close()
                        }
                        .addOnFailureListener(cameraExecutor) { e ->
                            Log.e(TAG, "QR detection failed", e)
                            if (cardCorners != null && cardCorners.size == 4) {
                                val scaleX = imageWidth / processWidth
                                val scaleY = imageHeight / processHeight
                                val screenCorners = cardCorners.map { Offset(it.x * scaleX, it.y * scaleY) }
                                coroutineScope.launch(Dispatchers.Main) {
                                    rawCardCorners = screenCorners
                                    isTracking = true
                                    rawQRCorners = null
                                    qrDetected = false
                                    rawBubblePositions = null
                                    rawBubbleStatus = null
                                    stableFrameCount = 0
                                    lastStableCorners = null
                                    autoCaptureTriggered = false
                                }
                            } else {
                                coroutineScope.launch(Dispatchers.Main) {
                                    isTracking = false
                                    rawCardCorners = null
                                    rawQRCorners = null
                                    rawBubblePositions = null
                                    rawBubbleStatus = null
                                    stableFrameCount = 0
                                    lastStableCorners = null
                                    autoCaptureTriggered = false
                                }
                            }
                            frameQueue.add(bitmap)
                            if (frameQueue.size > MAX_QUEUE) frameQueue.removeAt(0)
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
                    bubblePositions = smoothBubbles,
                    bubbleStatus = rawBubbleStatus,
                    isTracking = isTracking,
                    qrDetected = qrDetected,
                    scaleFactor = 1.08f
                )
                if (isScanning || isProcessing) {
                    ScanningLineOverlay(isScanning = true, status = scanStatus, frames = 1, totalFrames = 1)
                }
            }

            // ─── Status text ─────────────────────────────────────────
            val statusText = buildString {
                if (isTracking) append("✅ Card detected") else append("❌ Card not detected")
                if (qrDetected) append(" | QR ✓")
                else append(" | QR ✗")
                if (rawMarkerCenters != null) append(" | Markers ✓")
                else append(" | Markers ✗")
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
                        isScanning = true
                        scanStatus = "Capturing frames..."
                        isProcessing = true
                        coroutineScope.launch(Dispatchers.Default) {
                            delay(300)
                            val token = latestToken
                            val statuses = rawBubbleStatus
                            val groups = latestBubbleGroups

                            if (frameQueue.isEmpty() || token == null || statuses == null || groups == null) {
                                withContext(Dispatchers.Main) {
                                    isScanning = false
                                    isProcessing = false
                                    scanStatus = "Card not detected — hold steady and try again"
                                }
                                return@launch
                            }

                            val averaged = if (frameQueue.size >= 3) BubbleAnalyzer.averageBitmaps(frameQueue) else frameQueue.lastOrNull()
                            if (averaged == null) {
                                withContext(Dispatchers.Main) {
                                    isScanning = false
                                    isProcessing = false
                                    scanStatus = "No frame"
                                }
                                return@launch
                            }
                            val qrCorners = rawQRCorners?.map { PointF(it.x, it.y) }
                            val template = latestToken?.let { Templates.fromPrefix(it) }
                            if (qrCorners != null && template != null) {
                                val (filledIndices, confidence) = OMRExtractor.extractMarksWithConfidence(
                                    averaged,
                                    qrCorners,
                                    template
                                )
                                withContext(Dispatchers.Main) {
                                    scanResult = LocalScanResult(
                                        token = token,
                                        filledIndices = filledIndices,
                                        scanDataJson = Gson().toJson(mapOf("filledIndices" to filledIndices)),
                                        confidence = confidence
                                    )
                                    isScanning = false
                                    isProcessing = false
                                    scanStatus = "✅ Scan complete"
                                    frameQueue.clear()
                                }
                            } else {
                                withContext(Dispatchers.Main) {
                                    isScanning = false
                                    isProcessing = false
                                    scanStatus = "QR not detected"
                                }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(0.7f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFEB914))
                ) {
                    Text("Capture & Verify", color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }

            scanResult?.let { result ->
                AlertDialog(
                    onDismissRequest = { scanResult = null; isScanning = false },
                    title = { Text("Scan Result") },
                    text = {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("Token: ${result.token}", fontWeight = FontWeight.Bold)
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Confidence: ", fontWeight = FontWeight.Medium)
                                LinearProgressIndicator(
                                    progress = result.confidence,
                                    modifier = Modifier.weight(1f).height(8.dp).padding(horizontal = 8.dp),
                                    color = when {
                                        result.confidence > 0.8f -> Color(0xFF10B981)
                                        result.confidence > 0.5f -> Color(0xFFF59E0B)
                                        else -> Color(0xFFEF4444)
                                    },
                                    trackColor = Color.Gray.copy(alpha = 0.3f)
                                )
                                Text(
                                    text = "${String.format("%.0f", result.confidence * 100)}%",
                                    fontSize = 14.sp, fontWeight = FontWeight.Bold,
                                    color = when {
                                        result.confidence > 0.8f -> Color(0xFF10B981)
                                        result.confidence > 0.5f -> Color(0xFFF59E0B)
                                        else -> Color(0xFFEF4444)
                                    }
                                )
                            }
                            Text(
                                text = when {
                                    result.confidence > 0.8f -> "High confidence — reliable result."
                                    result.confidence > 0.5f -> "Medium confidence — consider rescanning."
                                    else -> "Low confidence — please rescan the card."
                                },
                                fontSize = 13.sp,
                                color = when {
                                    result.confidence > 0.8f -> Color(0xFF10B981)
                                    result.confidence > 0.5f -> Color(0xFFF59E0B)
                                    else -> Color(0xFFEF4444)
                                }
                            )
                            Divider()
                            Text("Data:", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(
                                result.scanDataJson,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                modifier = Modifier.background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)).padding(8.dp)
                            )
                        }
                    },
                    confirmButton = { Button(onClick = { scanResult = null; isScanning = false }) { Text(if (result.confidence > 0.5f) "OK" else "Rescan") } },
                    dismissButton = { TextButton(onClick = { scanResult = null; isScanning = false }) { Text("Cancel") } }
                )
            }
        }
    }
}

// ─── Coroutine Helper ──────────────────────────────────────────────
private suspend fun <T> Task<T>.await(): T {
    return suspendCancellableCoroutine { continuation ->
        addOnSuccessListener { result: T -> continuation.resume(result) }
        addOnFailureListener { exception: Exception -> continuation.resumeWithException(exception) }
    }
}