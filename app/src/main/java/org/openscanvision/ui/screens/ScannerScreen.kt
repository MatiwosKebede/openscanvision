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
import org.openscanvision.omr.CardDetector
import org.openscanvision.ui.components.StaticViewfinder
import java.util.concurrent.Executors

private const val TAG = "ArUcoScanner"

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
    // Standard clockwise-rotation point transform (matches how ImageInfo.rotationDegrees
    // is defined: degrees to rotate the buffer clockwise to appear upright).
    //   90°:  new image is (H x W). x' = H - y,        y' = x
    //   180°: new image is (W x H). x' = W - x,        y' = H - y
    //   270°: new image is (H x W). x' = y,             y' = W - x
    return when (rotationDegrees) {
        90 -> points.map { PointF(sensorHeight - it.y, it.x) }
        180 -> points.map { PointF(sensorWidth - it.x, sensorHeight - it.y) }
        270 -> points.map { PointF(it.y, sensorWidth - it.x) }
        else -> points
    }
}

/**
 * Maps points from a rotation-corrected analysis frame (frameWidth x frameHeight)
 * into PreviewView screen coordinates.
 *
 * This replicates PreviewView's default FILL_CENTER scaleType: uniform scale
 * (never stretched) that covers the view, then center-cropped. If your
 * PreviewView explicitly uses FIT_CENTER instead, change maxOf(...) to minOf(...).
 *
 * This is still an approximation. For pixel-perfect mapping in production,
 * prefer CameraX's androidx.camera.view.transform.OutputTransform +
 * CoordinateTransform, driven off a UseCaseGroup-bound ViewPort (see startCamera()
 * below) so Preview and ImageAnalysis observe identical crop rectangles.
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
    // Tracks whether previewView has completed at least one layout pass,
    // which is required before previewView.width/height/viewPort are valid.
    var previewLaidOut by remember { mutableStateOf(false) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    DisposableEffect(previewView) {
        val listener = android.view.View.OnLayoutChangeListener { _, _, _, _, _, oldL, oldT, oldR, oldB ->
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
                    val bitmap = imageProxy.toBitmap()
                    if (bitmap == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val bufferWidth = imageProxy.width.toFloat()
                    val bufferHeight = imageProxy.height.toFloat()

                    // Dimensions AFTER rotation is applied — swapped for 90/270.
                    val rotatedWidth = if (rotationDegrees % 180 == 0) bufferWidth else bufferHeight
                    val rotatedHeight = if (rotationDegrees % 180 == 0) bufferHeight else bufferWidth

                    // Detect ArUco markers in raw buffer space
                    val arUcoMap = CardDetector.detectArUcoMarkersFull(bitmap)
                    Log.d(TAG, "Detected ${arUcoMap.size} markers")

                    val screenMarkers = mutableMapOf<Int, Pair<Offset, List<Offset>>>()
                    for ((id, corners) in arUcoMap) {
                        // Buffer space -> rotation-corrected "upright" space
                        val sensorCorners = mapBitmapPointsToSensor(
                            corners, bufferWidth, bufferHeight, rotationDegrees
                        )
                        // Upright space -> PreviewView screen space
                        val screenCorners = mapImageToScreen(
                            sensorCorners.map { point -> Offset(point.x, point.y) },
                            rotatedWidth,
                            rotatedHeight,
                            previewView
                        )
                        if (screenCorners != null && screenCorners.size == 4) {
                            val centreX = screenCorners.map { it.x }.average().toFloat()
                            val centreY = screenCorners.map { it.y }.average().toFloat()
                            val centre = Offset(centreX, centreY)
                            screenMarkers[id] = Pair(centre, screenCorners)
                        }
                    }

                    coroutineScope.launch(Dispatchers.Main) {
                        detectedMarkers = screenMarkers
                        isTracking = screenMarkers.isNotEmpty()
                    }

                    imageProxy.close()
                }

                // Shared ViewPort so Preview and ImageAnalysis crop the sensor
                // identically. Without this, the two streams can show/analyze
                // different regions of the sensor even after the math above
                // is correct, causing a residual, hard-to-diagnose offset.
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
                    // Fallback: previewView hasn't laid out yet. Bind without a
                    // ViewPort so something is on screen, but coordinate mapping
                    // may be slightly off until the next recompose/restart after layout.
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

                // Draw marker outline (green polygon)
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

                // Draw centre dot (red)
                drawCircle(
                    center = centre,
                    radius = 8f,
                    color = Color.Red
                )

                // Draw ID and coordinates (white text with shadow)
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

                // Status bar
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