package org.openscanvision.core

import android.graphics.Bitmap
import android.graphics.PointF

sealed class ScanResult {
    data class Success(
        val token: String,
        val templateUsed: String,
        val filledIndices: List<Int>,   // 0‑based
        val confidence: Float,
        val json: String,
        val latencyMs: Long,
        val warpedBitmap: Bitmap? = null,
        val annotatedBitmap: Bitmap? = null,
        val qrPayload: String? = null,
        val qrCorners: List<PointF>? = null
    ) : ScanResult()

    sealed class Error : ScanResult() {
        object OpenCVNotInitialized : Error()
        object NoCardDetected : Error()
        object WarpFailed : Error()
        data class LowConfidence(val confidence: Float) : Error()
        data class TemplateMismatch(val detectedPrefix: String?) : Error()
        data class Generic(val message: String) : Error()
    }
}