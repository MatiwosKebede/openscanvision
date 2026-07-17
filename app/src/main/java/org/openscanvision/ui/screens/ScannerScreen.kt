package org.openscanvision.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import org.openscanvision.ui.components.CameraPreview

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val viewModel: ScannerViewModel = viewModel()
    val uiState by viewModel.uiState.collectAsState()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasCameraPermission = it
    }

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
        } else {
            CameraPreview(
                viewModel = viewModel,
                modifier = Modifier.fillMaxSize()
            )
            BottomHud(
                isTracking = uiState.isTracking,
                captureStatus = uiState.captureStatus,
                metricsLine = uiState.metricsLine,
                onForceScan = { viewModel.requestCapture() },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth(),
                brandGold = brandGold,
                textColor = darkBrown
            )
        }
    }

    // Show result dialog when scan result is available
    uiState.scanResultBitmap?.let { bitmap ->
        ScanResultDialog(
            omrBitmap = bitmap,
            templateName = uiState.templateName,
            confidence = uiState.confidence,
            filledIndices = uiState.filledIndices,
            voterId = uiState.voterId,
            voterName = uiState.voterName,
            qrToken = uiState.qrToken,
            onDismiss = { viewModel.dismissResult() },
            onAccept = { viewModel.dismissResult() },
            brandGold = brandGold,
            cardColor = white,
            background = white,
            textColor = darkBrown
        )
    }
}