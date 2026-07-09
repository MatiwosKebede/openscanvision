package org.openscanvision.model

import android.graphics.Bitmap

data class LocalScanResult(
    val token: String,
    val filledIndices: List<Int>,
    val scanDataJson: String,
    val confidence: Float = 1.0f,
    val warpedCardBitmap: Bitmap? = null,   // processed card
    val originalBitmap: Bitmap? = null      // original captured frame
)