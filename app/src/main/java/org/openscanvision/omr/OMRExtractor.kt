package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.util.Log

object OMRExtractor {
    private const val TAG = "OMRExtractor"
    private const val BUBBLE_RADIUS = 13   // in 0.1 mm units

    // Precomputed weighted-offset kernel (dx, dy, weight) for a given radius.
    // Built once per radius value instead of recomputed every bubble/frame.
    private val kernelCache = HashMap<Int, List<Triple<Int, Int, Float>>>()

    private fun kernelFor(radius: Int): List<Triple<Int, Int, Float>> =
        kernelCache.getOrPut(radius) {
            val pts = mutableListOf<Triple<Int, Int, Float>>()
            for (dy in -radius..radius) {
                for (dx in -radius..radius) {
                    val dist = kotlin.math.sqrt((dx * dx + dy * dy).toDouble())
                    if (dist > radius) continue
                    val weight = (1.0 - (dist / radius)).toFloat()
                    pts.add(Triple(dx, dy, weight))
                }
            }
            pts
        }

    fun extractCandidateMarks(
        originalBitmap: Bitmap,
        homography: Matrix
    ): Triple<List<Int>, Float, Bitmap?> {
        val warped = Bitmap.createBitmap(Templates.REF_WIDTH, Templates.REF_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(warped)
        canvas.drawColor(Color.WHITE)
        canvas.drawBitmap(originalBitmap, homography, Paint(Paint.FILTER_BITMAP_FLAG))

        var cleaned = ImagePreprocessor.enhanceContrast(warped)
        cleaned = ImagePreprocessor.denoise(cleaned)

        val (filled, confidence) = readCandidateBubbles(cleaned)
        Log.d(TAG, "Candidate (ArUco): filled=$filled, confidence=$confidence")
        return Triple(filled, confidence, cleaned)
    }

    fun extractCandidateMarks(
        originalBitmap: Bitmap,
        cardCornersImage: List<PointF>
    ): Triple<List<Int>, Float, Bitmap?> {
        if (cardCornersImage.size != 4) {
            Log.e(TAG, "Need exactly 4 card corners, got ${cardCornersImage.size}")
            return Triple(emptyList(), 0f, null)
        }
        val warped = OpenCVUtils.warpCard(originalBitmap, cardCornersImage, Templates.REF_WIDTH, Templates.REF_HEIGHT)
        if (warped == null) {
            Log.e(TAG, "OpenCV warp failed")
            return Triple(emptyList(), 0f, null)
        }
        var cleaned = ImagePreprocessor.enhanceContrast(warped)
        cleaned = ImagePreprocessor.denoise(cleaned)

        val (filled, confidence) = readCandidateBubbles(cleaned)
        Log.d(TAG, "Candidate (edge): filled=$filled, confidence=$confidence")
        return Triple(filled, confidence, cleaned)
    }

    fun extractAgendaMarks(
        originalBitmap: Bitmap,
        homography: Matrix
    ): Triple<List<Int>, Float, Bitmap?> {
        val warped = Bitmap.createBitmap(Templates.REF_WIDTH, Templates.REF_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(warped)
        canvas.drawColor(Color.WHITE)
        canvas.drawBitmap(originalBitmap, homography, Paint(Paint.FILTER_BITMAP_FLAG))

        var cleaned = ImagePreprocessor.enhanceContrast(warped)
        cleaned = ImagePreprocessor.denoise(cleaned)

        val (filled, confidence) = readAgendaBubbles(cleaned)
        Log.d(TAG, "Agenda (ArUco): filled=$filled, confidence=$confidence")
        return Triple(filled, confidence, cleaned)
    }

    // ─── Bubble reading (shared logic) — now on a single bulk pixel array ──

    private fun readCandidateBubbles(cleaned: Bitmap): Pair<List<Int>, Float> =
        readBubbles(cleaned, Templates.CANDIDATE.bubblePositions)

    private fun readAgendaBubbles(cleaned: Bitmap): Pair<List<Int>, Float> =
        readBubbles(cleaned, Templates.AGENDA.bubblePositions)

    private fun readBubbles(cleaned: Bitmap, positions: List<PointF>): Pair<List<Int>, Float> {
        val w = cleaned.width
        val h = cleaned.height
        // ONE bulk read instead of thousands of individual getPixel() JNI calls.
        val pixels = IntArray(w * h)
        cleaned.getPixels(pixels, 0, w, 0, 0, w, h)

        val kernelFull = kernelFor(BUBBLE_RADIUS)
        val kernelInner = kernelFor(BUBBLE_RADIUS / 2)

        val intensities = mutableListOf<Float>()
        val fillFlags = mutableListOf<Boolean>()

        for (pos in positions) {
            val cx = pos.x.toInt()
            val cy = pos.y.toInt()
            val full = sampleWeighted(pixels, w, h, cx, cy, kernelFull)
            val inner = sampleWeighted(pixels, w, h, cx, cy, kernelInner)
            val filledThreshold = 70f
            val isFilled = (full < filledThreshold) && (inner < filledThreshold * 0.9f)
            intensities.add(full)
            fillFlags.add(isFilled)
        }

        val (threshold, confidence) = ImagePreprocessor.adaptiveThresholdWithConfidence(
            intensities.map { it.toInt() }.toIntArray()
        )

        val filled = intensities.indices
            .filter { intensities[it] < threshold && fillFlags[it] }
            .toList()
        return Pair(filled, confidence)
    }

    private fun sampleWeighted(
        pixels: IntArray, width: Int, height: Int,
        cx: Int, cy: Int, kernel: List<Triple<Int, Int, Float>>
    ): Float {
        var sum = 0.0
        var weightSum = 0.0
        for ((dx, dy, weight) in kernel) {
            val x = cx + dx
            val y = cy + dy
            if (x < 0 || x >= width || y < 0 || y >= height) continue
            val pixel = pixels[y * width + x]
            // Grayscale already post-enhanceContrast, so R channel is representative.
            val gray = ((pixel shr 16) and 0xFF).toDouble()
            sum += gray * weight
            weightSum += weight
        }
        return if (weightSum > 0) (sum / weightSum).toFloat() else 255f
    }
}