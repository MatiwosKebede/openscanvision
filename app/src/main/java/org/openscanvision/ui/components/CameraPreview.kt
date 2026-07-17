package org.openscanvision.ui.components

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import org.openscanvision.ui.screens.ScannerViewModel
import java.util.concurrent.Executors

@Composable
fun CameraPreview(
    viewModel: ScannerViewModel,
    lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val previewView = remember { PreviewView(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    AndroidView(
        factory = { previewView },
        modifier = modifier,
        onRelease = {
            cameraExecutor.shutdown()
        }
    )

    DisposableEffect(previewView) {
        viewModel.setPreviewView(previewView)
        onDispose { }
    }

    LaunchedEffect(Unit) {
        val cameraProvider = getCameraProvider(context)
        try {
            cameraProvider.unbindAll()
            val preview = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 360))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setImageQueueDepth(1)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                viewModel.processFrame(imageProxy)
            }

            cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        } catch (e: Exception) {
            viewModel.setCameraError(e.message ?: "Camera error")
        }
    }
}

private suspend fun getCameraProvider(context: Context): ProcessCameraProvider =
    suspendCancellableCoroutine { cont ->
        ProcessCameraProvider.getInstance(context).addListener({
            cont.resume(ProcessCameraProvider.getInstance(context).get())
        }, ContextCompat.getMainExecutor(context))
    }