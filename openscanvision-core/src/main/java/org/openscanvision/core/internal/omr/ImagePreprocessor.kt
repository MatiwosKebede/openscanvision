package org.openscanvision.core.internal.omr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

internal object ImagePreprocessor {

    fun enhanceContrast(bitmap: Bitmap): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val destGray = Mat()
        clahe.apply(gray, destGray)

        val dest = Mat()
        Imgproc.cvtColor(destGray, dest, Imgproc.COLOR_GRAY2RGBA)
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dest, result)

        src.release()
        gray.release()
        destGray.release()
        dest.release()
        return result
    }

    fun denoise(bitmap: Bitmap): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }

        val denoised = Mat()
        Imgproc.medianBlur(gray, denoised, 3)

        val dest = Mat()
        Imgproc.cvtColor(denoised, dest, Imgproc.COLOR_GRAY2RGBA)
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dest, result)

        src.release()
        gray.release()
        denoised.release()
        dest.release()
        return result
    }

    fun normalize(bitmap: Bitmap): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }

        val normalized = Mat()
        org.opencv.core.Core.normalize(gray, normalized, 0.0, 255.0, org.opencv.core.Core.NORM_MINMAX)

        val dest = Mat()
        Imgproc.cvtColor(normalized, dest, Imgproc.COLOR_GRAY2RGBA)
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dest, result)

        src.release()
        gray.release()
        normalized.release()
        dest.release()
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
