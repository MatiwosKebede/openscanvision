package org.openscanvision.core.internal.omr.model

import android.graphics.Bitmap

internal data class LocalScanResult(
    val token: String,
    val filledIndices: List<Int>,
    val scanDataJson: String,
    val confidence: Float = 1.0f,
    val warpedCardBitmap: Bitmap? = null,
    val originalBitmap: Bitmap? = null
)
