package org.openscanvision.core.internal.omr

import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.sqrt

internal object BubbleAnalyzer {

    fun sampleDarkness(bitmap: Bitmap, positions: List<PointF>, radiusPx: Int): FloatArray {
        val result = FloatArray(positions.size)

        positions.forEachIndexed { index, pos ->
            val left = (pos.x - radiusPx).toInt().coerceIn(0, bitmap.width - 1)
            val top = (pos.y - radiusPx).toInt().coerceIn(0, bitmap.height - 1)
            val right = (pos.x + radiusPx).toInt().coerceIn(0, bitmap.width - 1)
            val bottom = (pos.y + radiusPx).toInt().coerceIn(0, bitmap.height - 1)
            val w = right - left
            val h = bottom - top

            if (w <= 0 || h <= 0) {
                result[index] = 0f
                return@forEachIndexed
            }

            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, left, top, w, h)

            var sum = 0L
            for (p in pixels) {
                sum += android.graphics.Color.red(p)
            }
            val avg = sum.toFloat() / pixels.size
            result[index] = 255f - avg   // invert: higher = darker
        }

        return result
    }

    fun classifyByGroup(
        darkness: FloatArray,
        groups: List<IntRange>,
        zThreshold: Double = 1.3
    ): BooleanArray {
        val marked = BooleanArray(darkness.size)

        for (group in groups) {
            val values = group.map { darkness[it] }
            val mean = values.average()
            val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
            val stdDev = sqrt(variance).coerceAtLeast(1.0)

            for (i in group) {
                val z = (darkness[i] - mean) / stdDev
                marked[i] = z > zThreshold
            }
        }

        return marked
    }

    fun computeConfidence(marked: BooleanArray, groups: List<IntRange>): Float {
        if (groups.isEmpty()) return 0f
        var goodGroups = 0
        for (group in groups) {
            val markedCount = group.count { marked[it] }
            if (markedCount == 1) goodGroups++
        }
        return goodGroups.toFloat() / groups.size
    }

    fun averageBitmaps(bitmaps: List<Bitmap>): Bitmap {
        require(bitmaps.isNotEmpty())
        val w = bitmaps.first().width
        val h = bitmaps.first().height
        val pixelCount = w * h

        val acc = IntArray(pixelCount)
        val buffer = IntArray(pixelCount)

        for (bmp in bitmaps) {
            bmp.getPixels(buffer, 0, w, 0, 0, w, h)
            for (i in 0 until pixelCount) {
                acc[i] += android.graphics.Color.red(buffer[i])
            }
        }

        val count = bitmaps.size
        val outPixels = IntArray(pixelCount)
        for (i in 0 until pixelCount) {
            val avg = acc[i] / count
            outPixels[i] = android.graphics.Color.rgb(avg, avg, avg)
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outPixels, 0, w, 0, 0, w, h)
        return result
    }
}
