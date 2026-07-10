package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log
import kotlin.math.abs

object OMRExtractor {
    private const val TAG = "OMRExtractor"
    private const val BUBBLE_RADIUS = 13   // in 0.1 mm units

    // Alignment quality thresholds (mean reprojection error, in image pixels)
    private const val REPROJ_ERROR_GOOD_PX = 3f
    private const val REPROJ_ERROR_REJECT_PX = 12f

    /**
     * Main extraction entry point.
     * When [cardCornersImage] is not null and contains 4 points, OpenCV warp is used.
     * Otherwise, falls back to the marker‑based homography.
     */
    fun extractCandidateMarks(
        originalBitmap: Bitmap,
        cardCornersImage: List<PointF>? = null,
        qrCorners: List<PointF> = emptyList()
    ): Triple<List<Int>, Float, Bitmap?> {
        var alignmentQuality: Float? = null

        val cleaned: Bitmap? = if (cardCornersImage != null && cardCornersImage.size == 4) {
            // Use OpenCV warp directly (requires OpenCVUtils in the same package)
            val warped = OpenCVUtils.warpCard(originalBitmap, cardCornersImage, Templates.REF_WIDTH, Templates.REF_HEIGHT)
            if (warped != null) {
                var c = ImagePreprocessor.enhanceContrast(warped)
                c = ImagePreprocessor.denoise(c)
                c
            } else {
                Log.e(TAG, "OpenCV warping failed, falling back to markers")
                val (bmp, quality) = warpAndCleanCandidateCardWithQuality(originalBitmap, qrCorners)
                alignmentQuality = quality
                bmp
            }
        } else {
            // Fallback: marker‑based homography
            val (bmp, quality) = warpAndCleanCandidateCardWithQuality(originalBitmap, qrCorners)
            alignmentQuality = quality
            bmp
        }

        if (cleaned == null) {
            Log.e(TAG, "Failed to obtain cleaned card image")
            return Triple(emptyList(), 0f, null)
        }

        val template = Templates.CANDIDATE
        val intensities = mutableListOf<Float>()
        val fillFlags = mutableListOf<Boolean>()

        for (pos in template.bubblePositions) {
            val (fullIntensity, isFilled) = sampleBubbleConcentric(
                cleaned, pos.x.toInt(), pos.y.toInt(), BUBBLE_RADIUS
            )
            intensities.add(fullIntensity)
            fillFlags.add(isFilled)
        }

        val (threshold, rawConfidence) = ImagePreprocessor.adaptiveThresholdWithConfidence(
            intensities.map { it.toInt() }.toIntArray()
        )

        val filled = intensities.indices
            .filter { intensities[it] < threshold && fillFlags[it] }
            .toList()

        // Alignment quality penalty only applies to marker fallback
        val alignmentFactor = alignmentQuality?.let { err ->
            ((REPROJ_ERROR_REJECT_PX - err) / (REPROJ_ERROR_REJECT_PX - REPROJ_ERROR_GOOD_PX)).coerceIn(0f, 1f)
        } ?: 1f

        val finalConfidence = (rawConfidence * alignmentFactor).coerceIn(0f, 1f)

        Log.d(TAG, "Candidate: filled=$filled, threshold=$threshold, rawConf=$rawConfidence, " +
                "alignErrPx=$alignmentQuality, finalConf=$finalConfidence")
        return Triple(filled, finalConfidence, cleaned)
    }

    // ─── Fallback marker‑based warping ──────────────────────────────

    private fun warpAndCleanCandidateCardWithQuality(
        originalBitmap: Bitmap,
        qrCorners: List<PointF>
    ): Pair<Bitmap?, Float?> {
        val template = Templates.CANDIDATE
        val (homography, quality) = computeMarkerHomographyWithQuality(originalBitmap, qrCorners, template)
            ?: return Pair(null, null)

        val warped = Bitmap.createBitmap(Templates.REF_WIDTH, Templates.REF_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(warped)
        canvas.drawColor(Color.WHITE)
        canvas.drawBitmap(originalBitmap, homography, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))

        var cleaned = ImagePreprocessor.enhanceContrast(warped)
        cleaned = ImagePreprocessor.denoise(cleaned)
        cleaned = ImagePreprocessor.normalize(cleaned)
        warped.recycle()
        return Pair(cleaned, quality)
    }

    fun computeMarkerHomography(
        originalBitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Matrix? = computeMarkerHomographyWithQuality(originalBitmap, qrCorners, template)?.first

    fun computeMarkerHomographyWithQuality(
        originalBitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Pair<Matrix, Float>? {
        if (qrCorners.size != 4) return null

        var homography = HomographySolver.solve(template.qrRefCorners, qrCorners) ?: run {
            Log.e(TAG, "Initial QR-only homography failed")
            return null
        }

        val markerRefs = template.markerRefPositions ?: Templates.SHARED_MARKER_CORNERS
        var bestHomography = homography
        var bestQuality = Float.MAX_VALUE

        repeat(2) { pass ->
            val predicted = CardDetector.predictImagePoints(markerRefs, homography)
            val detected = CardDetector.detectRefMarkersNearPredicted(originalBitmap, predicted)

            val combinedTemplate = mutableListOf<PointF>().apply { addAll(template.qrRefCorners) }
            val combinedImage = mutableListOf<PointF>().apply { addAll(qrCorners) }

            for (i in markerRefs.indices) {
                val det = detected.getOrNull(i) ?: continue
                // Sub‑pixel refinement on the dark marker blob
                val refined = refineMarkerCenterSubpixel(originalBitmap, det, windowRadius = 12)
                combinedTemplate.add(markerRefs[i])
                combinedImage.add(refined)
            }

            val result = computeRobustHomography(combinedTemplate, combinedImage, maxReprojErrorPx = 6f)
            if (result != null) {
                val (refinedHomography, quality) = result
                homography = refinedHomography
                if (quality < bestQuality) {
                    bestQuality = quality
                    bestHomography = refinedHomography
                }
                Log.d(TAG, "Homography refine pass $pass: pts=${combinedTemplate.size}, meanErrPx=$quality")
            } else {
                Log.w(TAG, "Homography refine pass $pass produced no valid fit")
            }
        }

        if (bestQuality == Float.MAX_VALUE) {
            Log.w(TAG, "Falling back to QR-only homography")
            return Pair(bestHomography, REPROJ_ERROR_REJECT_PX - 0.01f)
        }

        if (bestQuality > REPROJ_ERROR_REJECT_PX) {
            Log.w(TAG, "Homography rejected: meanErrorPx=$bestQuality")
            return null
        }

        return Pair(bestHomography, bestQuality)
    }

    /** Refines a dark marker centre to sub‑pixel precision using intensity‑weighted centroid. */
    private fun refineMarkerCenterSubpixel(
        bitmap: Bitmap, approx: PointF, windowRadius: Int = 8
    ): PointF {
        val width = bitmap.width; val height = bitmap.height
        val cx0 = approx.x.toInt(); val cy0 = approx.y.toInt()

        var sumX = 0.0; var sumY = 0.0; var sumWeight = 0.0

        for (dy in -windowRadius..windowRadius) {
            for (dx in -windowRadius..windowRadius) {
                val x = cx0 + dx; val y = cy0 + dy
                if (x < 0 || x >= width || y < 0 || y >= height) continue
                val pixel = bitmap.getPixel(x, y)
                val gray = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3.0
                val weight = (255.0 - gray).coerceAtLeast(0.0)
                sumX += x * weight; sumY += y * weight; sumWeight += weight
            }
        }
        return if (sumWeight > 1e-3) {
            PointF((sumX / sumWeight).toFloat(), (sumY / sumWeight).toFloat())
        } else approx
    }

    /**
     * Robust homography fit with iterative worst‑point removal.
     * Returns (Matrix, meanReprojectionError) or null.
     */
    private fun computeRobustHomography(
        templatePoints: List<PointF>,
        imagePoints: List<PointF>,
        maxReprojErrorPx: Float,
        minPoints: Int = 4
    ): Pair<Matrix, Float>? {
        if (templatePoints.size != imagePoints.size || templatePoints.size < minPoints) return null

        val curTmpl = templatePoints.toMutableList()
        val curImg = imagePoints.toMutableList()
        var homography = HomographySolver.solve(curTmpl, curImg) ?: return null

        while (curTmpl.size > minPoints) {
            val errors = HomographySolver.reprojectionErrors(homography, curTmpl, curImg)
            val maxErr = errors.maxOrNull() ?: break
            if (maxErr <= maxReprojErrorPx) break

            val worstIdx = errors.indexOf(maxErr)
            curTmpl.removeAt(worstIdx)
            curImg.removeAt(worstIdx)
            val refit = HomographySolver.solve(curTmpl, curImg) ?: break
            homography = refit
        }

        val finalErrors = HomographySolver.reprojectionErrors(homography, curTmpl, curImg)
        val meanErr = if (finalErrors.isNotEmpty()) finalErrors.average().toFloat() else Float.MAX_VALUE
        return Pair(homography, meanErr)
    }

    // ─── Concentric bubble sampling ─────────────────────────────────

    private fun sampleBubbleConcentric(
        bitmap: Bitmap,
        cx: Int, cy: Int,
        radius: Int
    ): Pair<Float, Boolean> {
        val full = sampleBubbleWeighted(bitmap, cx, cy, radius)
        val inner = sampleBubbleWeighted(bitmap, cx, cy, radius / 2)

        val filledThreshold = 70f
        val isFilled = (full < filledThreshold) && (inner < filledThreshold * 0.9f)
        return Pair(full, isFilled)
    }

    private fun sampleBubbleWeighted(bitmap: Bitmap, cx: Int, cy: Int, radius: Int): Float {
        val width = bitmap.width; val height = bitmap.height
        var sum = 0.0; var weightSum = 0.0

        for (dy in -radius..radius) {
            for (dx in -radius..radius) {
                val x = cx + dx; val y = cy + dy
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

// ─── Homography solver (unchanged) ───
private object HomographySolver {

    fun solve(src: List<PointF>, dst: List<PointF>): Matrix? {
        if (src.size != dst.size || src.size < 4) return null

        val (srcNorm, tSrcForward) = normalizePoints(src) ?: return null
        val (dstNorm, tDstForward) = normalizePoints(dst) ?: return null

        val n = srcNorm.size
        val a = Array(2 * n) { DoubleArray(8) }
        val b = DoubleArray(2 * n)

        for (i in 0 until n) {
            val x = srcNorm[i].x.toDouble()
            val y = srcNorm[i].y.toDouble()
            val xp = dstNorm[i].x.toDouble()
            val yp = dstNorm[i].y.toDouble()

            a[2 * i][0] = x; a[2 * i][1] = y; a[2 * i][2] = 1.0
            a[2 * i][3] = 0.0; a[2 * i][4] = 0.0; a[2 * i][5] = 0.0
            a[2 * i][6] = -x * xp; a[2 * i][7] = -y * xp
            b[2 * i] = xp

            a[2 * i + 1][0] = 0.0; a[2 * i + 1][1] = 0.0; a[2 * i + 1][2] = 0.0
            a[2 * i + 1][3] = x; a[2 * i + 1][4] = y; a[2 * i + 1][5] = 1.0
            a[2 * i + 1][6] = -x * yp; a[2 * i + 1][7] = -y * yp
            b[2 * i + 1] = yp
        }

        val h = solveLeastSquares(a, b, 2 * n, 8) ?: return null

        val hNorm = Matrix()
        hNorm.setValues(
            floatArrayOf(
                h[0].toFloat(), h[1].toFloat(), h[2].toFloat(),
                h[3].toFloat(), h[4].toFloat(), h[5].toFloat(),
                h[6].toFloat(), h[7].toFloat(), 1f
            )
        )

        val tDstInv = Matrix()
        if (!tDstForward.invert(tDstInv)) return null

        val result = Matrix(tDstInv)
        result.preConcat(hNorm)
        result.preConcat(tSrcForward)
        return result
    }

    fun reprojectionErrors(homography: Matrix, templatePoints: List<PointF>, imagePoints: List<PointF>): List<Float> {
        val srcArray = FloatArray(templatePoints.size * 2)
        templatePoints.forEachIndexed { i, pt ->
            srcArray[i * 2] = pt.x
            srcArray[i * 2 + 1] = pt.y
        }
        val mapped = FloatArray(srcArray.size)
        homography.mapPoints(mapped, srcArray)

        return imagePoints.indices.map { i ->
            val dx = mapped[i * 2] - imagePoints[i].x
            val dy = mapped[i * 2 + 1] - imagePoints[i].y
            kotlin.math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
        }
    }

    private fun normalizePoints(points: List<PointF>): Pair<List<PointF>, Matrix>? {
        val cx = points.map { it.x }.average().toFloat()
        val cy = points.map { it.y }.average().toFloat()
        val meanDist = points.map {
            kotlin.math.sqrt(((it.x - cx) * (it.x - cx) + (it.y - cy) * (it.y - cy)).toDouble())
        }.average()

        if (meanDist < 1e-6) return null

        val scale = (kotlin.math.sqrt(2.0) / meanDist).toFloat()

        val transform = Matrix().apply {
            postTranslate(-cx, -cy)
            postScale(scale, scale)
        }

        val srcArray = FloatArray(points.size * 2)
        points.forEachIndexed { i, pt ->
            srcArray[i * 2] = pt.x
            srcArray[i * 2 + 1] = pt.y
        }
        val mapped = FloatArray(srcArray.size)
        transform.mapPoints(mapped, srcArray)

        val normalized = points.indices.map { i -> PointF(mapped[i * 2], mapped[i * 2 + 1]) }
        return Pair(normalized, transform)
    }

    private fun solveLeastSquares(a: Array<DoubleArray>, b: DoubleArray, rows: Int, cols: Int): DoubleArray? {
        val ata = Array(cols) { DoubleArray(cols) }
        val atb = DoubleArray(cols)

        for (i in 0 until cols) {
            for (j in 0 until cols) {
                var sum = 0.0
                for (k in 0 until rows) sum += a[k][i] * a[k][j]
                ata[i][j] = sum
            }
            var sumB = 0.0
            for (k in 0 until rows) sumB += a[k][i] * b[k]
            atb[i] = sumB
        }

        return gaussianSolve(ata, atb, cols)
    }

    private fun gaussianSolve(a: Array<DoubleArray>, b: DoubleArray, n: Int): DoubleArray? {
        val aug = Array(n) { i ->
            DoubleArray(n + 1).also { row ->
                for (j in 0 until n) row[j] = a[i][j]
                row[n] = b[i]
            }
        }

        for (col in 0 until n) {
            var pivotRow = col
            var maxVal = abs(aug[col][col])
            for (r in col + 1 until n) {
                if (abs(aug[r][col]) > maxVal) {
                    maxVal = abs(aug[r][col])
                    pivotRow = r
                }
            }
            if (maxVal < 1e-10) return null

            val temp = aug[col]
            aug[col] = aug[pivotRow]
            aug[pivotRow] = temp

            for (r in 0 until n) {
                if (r == col) continue
                val factor = aug[r][col] / aug[col][col]
                for (c in col until n + 1) {
                    aug[r][c] -= factor * aug[col][c]
                }
            }
        }

        return DoubleArray(n) { i -> aug[i][n] / aug[i][i] }
    }
}