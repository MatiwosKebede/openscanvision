package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import android.widget.Toast
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

private fun mapImageToScreen(
    points: List<Offset>,
    sensorWidth: Float,
    sensorHeight: Float,
    previewView: PreviewView
): List<Offset>? {
    // Simple proportional mapping (assumes full screen, no crop)
    // In your real app, use CameraX's proper coordinate mapping.
    // This is a placeholder that works for full-screen previews.
    val viewWidth = previewView.width.toFloat()
    val viewHeight = previewView.height.toFloat()
    if (viewWidth <= 0 || viewHeight <= 0) return null
    val scaleX = viewWidth / sensorWidth
    val scaleY = viewHeight / sensorHeight
    return points.map { Offset(it.x * scaleX, it.y * scaleY) }
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

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
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
                    val mediaImage = imageProxy.image ?: run { imageProxy.close(); return@setAnalyzer }
                    val bitmap = imageProxy.toBitmap() ?: run { imageProxy.close(); return@setAnalyzer }
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val sensorWidth = imageProxy.width.toFloat()
                    val sensorHeight = imageProxy.height.toFloat()

                    // Detect ArUco markers
                    val arUcoMap = CardDetector.detectArUcoMarkersFull(bitmap)
                    Log.d(TAG, "Detected ${arUcoMap.size} markers")

                    // Map corners to screen coordinates
                    val screenMarkers = mutableMapOf<Int, Pair<Offset, List<Offset>>>()
                    for ((id, corners) in arUcoMap) {
                        // Convert corners from bitmap space to sensor space
                        val sensorCorners = mapBitmapPointsToSensor(corners, sensorWidth, sensorHeight, rotationDegrees)
                        // Convert sensor corners to screen coordinates
                        val screenCorners = mapImageToScreen(
                            sensorCorners.map { point -> Offset(point.x, point.y) },
                            sensorWidth,
                            sensorHeight,
                            previewView
                        )
                        if (screenCorners != null && screenCorners.size == 4) {
                            val centreX = screenCorners.map { it.x }.average().toFloat()
                            val centreY = screenCorners.map { it.y }.average().toFloat()
                            val centre = Offset(centreX, centreY)
                            screenMarkers[id] = Pair(centre, screenCorners)
                        }
                    }

                    // Update UI
                    coroutineScope.launch(Dispatchers.Main) {
                        detectedMarkers = screenMarkers
                        isTracking = screenMarkers.isNotEmpty()
                    }

                    imageProxy.close()
                }

                cameraProvider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
                cameraError = null
                Log.d(TAG, "Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera start error", e)
                cameraError = "Failed to start camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }

    LaunchedEffect(hasCameraPermission) {
        if (hasCameraPermission) startCamera() else permissionLauncher.launch(Manifest.permission.CAMERA)
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