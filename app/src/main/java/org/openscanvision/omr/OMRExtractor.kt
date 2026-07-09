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

        // 2. Perspective correction (Initial alignment via QR)
        val refWidth = Templates.REF_WIDTH
        val refHeight = Templates.REF_HEIGHT
        val warped = warpPerspective(processed, qrCorners, template, refWidth, refHeight)
        if (warped == null) {
            Log.e(TAG, "Perspective correction failed")
            return Pair(emptyList(), 0f)
        }

        // 3. Localize the 4 square reference marks on the warped image for micro-precision refinement
        val theoreticalMarkers = template.markerRefPositions ?: Templates.SHARED_MARKER_CORNERS
        val detectedMarkers = CardDetector.detectRefMarkersNearPredicted(warped, theoreticalMarkers)

        val validTemplatePoints = mutableListOf<PointF>()
        val validImagePoints = mutableListOf<PointF>()

        for (i in theoreticalMarkers.indices) {
            val det = detectedMarkers.getOrNull(i)
            if (det != null) {
                validTemplatePoints.add(theoreticalMarkers[i])
                validImagePoints.add(det)
            }
        }

        // Generate micro-alignment correction mapping matrix if all 4 corner blocks are parsed
        val refinementMatrix = if (validTemplatePoints.size >= 4) {
            CardDetector.buildRefinedHomography(validTemplatePoints, validImagePoints)
        } else {
            Log.w(TAG, "Could not find all 4 square markers, falling back to basic QR mapping.")
            null
        }

        // 4. Extract bubble intensities using the refined coordinate space
        val allIntensities = mutableListOf<Int>()
        val bubbleResults = mutableListOf<Pair<Int, Float>>()
        val tempCoords = FloatArray(2)

        for ((index, pos) in template.bubblePositions.withIndex()) {
            val finalX: Int
            val finalY: Int

            if (refinementMatrix != null) {
                tempCoords[0] = pos.x
                tempCoords[1] = pos.y
                refinementMatrix.mapPoints(tempCoords)
                finalX = tempCoords[0].toInt()
                finalY = tempCoords[1].toInt()
            } else {
                finalX = pos.x.toInt()
                finalY = pos.y.toInt()
            }

            val avgIntensity = sampleBubbleWeighted(warped, finalX, finalY, BUBBLE_RADIUS)
            allIntensities.add(avgIntensity.toInt())
            bubbleResults.add(Pair(index, avgIntensity))
        }

        // 5. Otsu threshold with confidence
        val (threshold, confidence) = ImagePreprocessor.adaptiveThresholdWithConfidence(
            allIntensities.toIntArray()
        )

        // 6. Determine filled bubbles (dark = intensity < threshold)
        val filledIndices = mutableListOf<Int>()
        for ((index, intensity) in bubbleResults) {
            if (intensity < threshold) {
                filledIndices.add(index)
            }
        }

        // 7. Adjust confidence based on borderline bubbles
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