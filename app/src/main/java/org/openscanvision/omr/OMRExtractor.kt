// OMRExtractor.kt
package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log

object OMRExtractor {
    private const val TAG = "OMRExtractor"
    private const val BUBBLE_RADIUS = 20

    fun extractMarksWithConfidence(
        bitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Pair<List<Int>, Float> {
        if (qrCorners.size != 4) {
            Log.e(TAG, "QR corners must be exactly 4 points")
            return Pair(emptyList(), 0f)
        }

        // 1. Pre‑process
        var processed = ImagePreprocessor.enhanceContrast(bitmap)
        processed = ImagePreprocessor.denoise(processed)
        processed = ImagePreprocessor.normalize(processed)

        // 2. Perspective correction
        val refWidth = Templates.REF_WIDTH
        val refHeight = Templates.REF_HEIGHT
        val warped = warpPerspective(processed, qrCorners, template, refWidth, refHeight)
        if (warped == null) {
            Log.e(TAG, "Perspective correction failed")
            return Pair(emptyList(), 0f)
        }

        // 3. Extract bubble intensities
        val allIntensities = mutableListOf<Int>()
        val bubbleResults = mutableListOf<Pair<Int, Float>>()

        for ((index, pos) in template.bubblePositions.withIndex()) {
            val x = pos.x.toInt()
            val y = pos.y.toInt()
            val avgIntensity = sampleBubbleWeighted(warped, x, y, BUBBLE_RADIUS)
            allIntensities.add(avgIntensity.toInt())
            bubbleResults.add(Pair(index, avgIntensity))
        }

        // 4. Otsu threshold with confidence
        val (threshold, confidence) = ImagePreprocessor.adaptiveThresholdWithConfidence(
            allIntensities.toIntArray()
        )

        // 5. Determine filled bubbles (dark = intensity < threshold)
        val filledIndices = mutableListOf<Int>()
        for ((index, intensity) in bubbleResults) {
            if (intensity < threshold) {
                filledIndices.add(index)
            }
        }

        // 6. Adjust confidence based on borderline bubbles
        val margin = threshold * 0.15f
        var lowConfidenceCount = 0
        for ((_, intensity) in bubbleResults) {
            if (intensity in (threshold - margin)..(threshold + margin)) {
                lowConfidenceCount++
            }
        }
        val adjustedConfidence = confidence * (1f - (lowConfidenceCount * 0.05f))

        Log.d(TAG, "Filled: $filledIndices, threshold=$threshold, confidence=$adjustedConfidence")

        processed.recycle()
        warped.recycle()

        return Pair(filledIndices, adjustedConfidence)
    }

    private fun warpPerspective(
        bitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate,
        refWidth: Int,
        refHeight: Int
    ): Bitmap? {
        val matrix = Matrix()
        val srcPoints = FloatArray(8)
        val dstPoints = FloatArray(8)

        for (i in 0..3) {
            srcPoints[i * 2] = qrCorners[i].x
            srcPoints[i * 2 + 1] = qrCorners[i].y
            dstPoints[i * 2] = template.qrRefCorners[i].x
            dstPoints[i * 2 + 1] = template.qrRefCorners[i].y
        }

        if (!matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 4)) {
            Log.e(TAG, "setPolyToPoly failed")
            return null
        }

        val warped = Bitmap.createBitmap(refWidth, refHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(warped)
        canvas.drawColor(Color.WHITE)
        canvas.drawBitmap(bitmap, matrix, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
        return warped
    }

    private fun sampleBubbleWeighted(bitmap: Bitmap, cx: Int, cy: Int, radius: Int): Float {
        val width = bitmap.width
        val height = bitmap.height
        var sum = 0.0
        var weightSum = 0.0

        for (dy in -radius..radius) {
            for (dx in -radius..radius) {
                val x = cx + dx
                val y = cy + dy
                if (x < 0 || x >= width || y < 0 || y >= height) continue
                val dist = kotlin.math.sqrt((dx * dx + dy * dy).toDouble())
                if (dist > radius) continue
                val weight = 1.0 - (dist / radius)
                val pixel = bitmap.getPixel(x, y)
                val gray = Color.red(pixel).toDouble()
                sum += gray * weight
                weightSum += weight
            }
        }
        return (sum / weightSum).toFloat()
    }
}