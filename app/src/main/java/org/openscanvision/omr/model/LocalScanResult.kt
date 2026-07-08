package org.openscanvision.model

data class LocalScanResult(
    val token: String,
    val filledIndices: List<Int>,
    val scanDataJson: String,
    val confidence: Float = 1.0f
)