// ImagePreprocessor.kt
package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Color

object ImagePreprocessor {

    fun enhanceContrast(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val gray = FloatArray(pixels.size)
        val histogram = IntArray(256)
        for (i in pixels.indices) {
            val r = Color.red(pixels[i])
            val g = Color.green(pixels[i])
            val b = Color.blue(pixels[i])
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
            histogram[gray[i].toInt()]++
        }

        val cdf = FloatArray(256)
        var sum = 0
        for (i in 0..255) {
            sum += histogram[i]
            cdf[i] = sum.toFloat() / pixels.size
        }

        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val newGray = (cdf[gray[i].toInt()] * 255).toInt()
            result.setPixel(i % width, i / width, Color.rgb(newGray, newGray, newGray))
        }
        return result
    }

    fun denoise(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val kernel = IntArray(9)

        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var idx = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val px = x + dx
                        val py = y + dy
                        kernel[idx++] = Color.red(pixels[py * width + px])
                    }
                }
                kernel.sort()
                val median = kernel[4]
                result.setPixel(x, y, Color.rgb(median, median, median))
            }
        }
        return result
    }

    fun normalize(bitmap: Bitmap): Bitmap {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var min = 255
        var max = 0
        for (p in pixels) {
            val g = Color.red(p)
            if (g < min) min = g
            if (g > max) max = g
        }
        val range = (max - min).toFloat()
        if (range < 10) return bitmap

        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val g = Color.red(pixels[i])
            val normalized = ((g - min) / range * 255).toInt()
            result.setPixel(i % bitmap.width, i / bitmap.width, Color.rgb(normalized, normalized, normalized))
        }
        return result
    }

    fun otsuThreshold(intensities: IntArray): Int {
        if (intensities.isEmpty()) return 128

        val histogram = IntArray(256)
        for (value in intensities) {
            histogram[value.coerceIn(0, 255)]++
        }

        val total = intensities.size
        var sum = 0.0
        for (i in 0..255) {
            sum += i * histogram[i]
        }

        var sumB = 0.0
        var wB = 0
        var wF = 0
        var varMax = 0.0
        var threshold = 128

        for (i in 0..255) {
            wB += histogram[i]
            if (wB == 0) continue
            wF = total - wB
            if (wF == 0) break

            sumB += i * histogram[i]
            val mB = sumB / wB
            val mF = (sum - sumB) / wF
            val varBetween = wB.toDouble() * wF.toDouble() * (mB - mF) * (mB - mF)

            if (varBetween > varMax) {
                varMax = varBetween
                threshold = i
            }
        }
        return threshold
    }

    fun adaptiveThresholdWithConfidence(intensities: IntArray): Pair<Int, Float> {
        val threshold = otsuThreshold(intensities)

        val above = intensities.count { it >= threshold }
        val below = intensities.count { it < threshold }
        if (above == 0 || below == 0) return Pair(threshold, 0.5f)

        val aboveMean = intensities.filter { it >= threshold }.average()
        val belowMean = intensities.filter { it < threshold }.average()
        val separation = (aboveMean - belowMean).toFloat() / 255f
        val confidence = separation.coerceIn(0f, 1f)

        return Pair(threshold, confidence)
    }
}