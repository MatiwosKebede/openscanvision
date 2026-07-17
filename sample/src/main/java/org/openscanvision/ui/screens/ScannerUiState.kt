package org.openscanvision.ui.screens

import android.graphics.Bitmap

data class ScannerUiState(
    val isTracking: Boolean = false,
    val captureStatus: String = "Position the card inside the viewfinder to scan.",
    val metricsLine: String = "Attempts: 0 | Success: 0 | Avg lock: 0f | Avg time: 0ms | Reject: 0.0%",
    val scanResultBitmap: Bitmap? = null,
    val filledIndices: List<Int> = emptyList(),
    val confidence: Float = 0f,
    val templateName: String? = null,
    val qrToken: String? = null,
    val voterId: String = "",
    val voterName: String = ""
)