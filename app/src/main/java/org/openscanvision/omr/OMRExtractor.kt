package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log

object OMRExtractor {
    private const val TAG = "OMRExtractor"
    private const val BUBBLE_RADIUS = 13   // in 0.1mm units

    /**
     * Directly warp and clean a bitmap using a pre‑computed homography.
     * This ensures the processed image uses the same transformation as the live overlay.
     */
    fun warpWithHomography(bitmap: Bitmap, homography: Matrix): Bitmap? {
        return try {
            val warped = Bitmap.createBitmap(Templates.REF_WIDTH, Templates.REF_HEIGHT, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(warped)
            canvas.drawColor(Color.WHITE)
            canvas.drawBitmap(bitmap, homography, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))

            var cleaned = ImagePreprocessor.enhanceContrast(warped)
            cleaned = ImagePreprocessor.denoise(cleaned)
            cleaned = ImagePreprocessor.normalize(cleaned)
            warped.recycle()
            cleaned
        } catch (e: Exception) {
            Log.e(TAG, "warpWithHomography error", e)
            null
        }
    }

    /**
     * Main entry point for Candidate card.
     * If a homography is provided, it is used directly; otherwise, it is recomputed from QR corners.
     */
    fun extractCandidateMarks(
        originalBitmap: Bitmap,
        qrCorners: List<PointF> = emptyList(),
        homography: Matrix? = null
    ): Triple<List<Int>, Float, Bitmap?> {
        val cleaned = if (homography != null) {
            warpWithHomography(originalBitmap, homography)
        } else {
            warpAndCleanCandidateCard(originalBitmap, qrCorners)
        }

        if (cleaned == null) {
            Log.e(TAG, "Failed to warp card")
            return Triple(emptyList(), 0f, null)
        }

        val template = Templates.CANDIDATE
        val intensities = mutableListOf<Float>()
        for (pos in template.bubblePositions) {
            val intensity = sampleBubbleWeighted(cleaned, pos.x.toInt(), pos.y.toInt(), BUBBLE_RADIUS)
            intensities.add(intensity)
        }

        val (threshold, confidence) = ImagePreprocessor.adaptiveThresholdWithConfidence(
            intensities.map { it.toInt() }.toIntArray()
        )

        val filled = intensities.mapIndexedNotNull { idx, value ->
            if (value < threshold) idx else null
        }

        Log.d(TAG, "Candidate: filled=$filled, threshold=$threshold, confidence=$confidence")
        return Triple(filled, confidence, cleaned)
    }

    /**
     * Warps the original bitmap to template size using homography from 4 square markers.
     * Returns a cleaned, upright, standardised bitmap.
     */
    private fun warpAndCleanCandidateCard(
        originalBitmap: Bitmap,
        qrCorners: List<PointF>
    ): Bitmap? {
        val template = Templates.CANDIDATE
        val homography = computeMarkerHomography(originalBitmap, qrCorners, template)
            ?: return null

        val warped = Bitmap.createBitmap(Templates.REF_WIDTH, Templates.REF_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(warped)
        canvas.drawColor(Color.WHITE)
        canvas.drawBitmap(originalBitmap, homography, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))

        var cleaned = ImagePreprocessor.enhanceContrast(warped)
        cleaned = ImagePreprocessor.denoise(cleaned)
        cleaned = ImagePreprocessor.normalize(cleaned)
        warped.recycle()
        return cleaned
    }

    /**
     * Computes homography directly from template markers to detected image markers.
     * Uses QR homography to predict marker positions, then refines with actual detection.
     * If not enough markers are found, falls back to QR-only homography.
     */
    fun computeMarkerHomography(
        originalBitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Matrix? {
        // 1. Build QR homography (always needed as fallback)
        val qrMatrix = Matrix()
        val srcPoints = FloatArray(8)
        val dstPoints = FloatArray(8)
        for (i in 0..3) {
            srcPoints[i*2] = template.qrRefCorners[i].x
            srcPoints[i*2+1] = template.qrRefCorners[i].y
            dstPoints[i*2] = qrCorners[i].x
            dstPoints[i*2+1] = qrCorners[i].y
        }
        if (!qrMatrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 4)) {
            Log.e(TAG, "QR homography failed")
            return null
        }

        // 2. Predict markers
        val markerRefs = template.markerRefPositions ?: Templates.SHARED_MARKER_CORNERS
        val predicted = CardDetector.predictImagePoints(markerRefs, qrMatrix)

        // 3. Detect markers on original bitmap
        val detected = CardDetector.detectRefMarkersNearPredicted(originalBitmap, predicted)

        // 4. Build correspondences
        val validTemplate = mutableListOf<PointF>()
        val validImage = mutableListOf<PointF>()
        for (i in markerRefs.indices) {
            val det = detected.getOrNull(i)
            if (det != null) {
                validTemplate.add(markerRefs[i])
                validImage.add(det)
            }
        }

        // 5. If less than 4 markers, fall back to QR homography
        if (validTemplate.size < 4) {
            Log.w(TAG, "Only ${validTemplate.size} markers, falling back to QR homography")
            return qrMatrix
        }

        // 6. Build refined homography from markers
        return CardDetector.buildRefinedHomography(validTemplate, validImage)
    }

    /**
     * Weighted average of pixel intensities inside a circular region.
     */
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