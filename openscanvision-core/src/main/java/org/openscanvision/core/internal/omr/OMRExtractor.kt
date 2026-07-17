package org.openscanvision.core.internal.omr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

object OMRExtractor {
    private const val TAG = "OMRExtractor"

    // ── Geometry ──────────────────────────────────────────────
    private const val BUBBLE_RADIUS = 10
    private const val RING_INNER = 19
    private const val RING_OUTER = 27
    private const val INNER_FILL_RADIUS = 6
    private const val REFINE_SEARCH_RADIUS = 4
    private const val REFINE_SEARCH_STEP = 2

    // ── Decision thresholds (balanced) ────────────────────────
    private const val Z_THRESHOLD = 1.3f                // was 1.8 – more sensitive
    private const val MIN_ABSOLUTE_DARKNESS = 0.25f     // was 0.35 – lower barrier

    // Ink classification
    private const val INK_RELATIVE_DROP = 0.32f         // was 0.40
    private const val INK_MIN_ABSOLUTE_DROP = 30f       // was 40

    private const val CENTROID_REFINE_MIN_SIGNAL = 8f
    private const val ENABLE_ILLUMINATION_FLATTENING = false  // disable if artifacts

    // Weighting (more weight on darkness)
    private const val DARKNESS_WEIGHT = 0.6f
    private const val FILL_WEIGHT = 0.4f

    enum class GroupStatus { EMPTY, SINGLE, OVERVOTE, LOW_CONFIDENCE }

    data class BubbleRead(
        val index: Int,
        val normalizedDarkness: Float,
        val rawIntensity: Float,
        val localBackground: Float,
        val fillRatio: Float = 0f
    )

    data class GroupResult(
        val range: IntRange,
        val status: GroupStatus,
        val filledIndices: List<Int>,
        val confidence: Float,
        val reads: List<BubbleRead>
    )

    data class BubbleReport(
        val groups: List<GroupResult>,
        val allFilled: List<Int>,
        val overallConfidence: Float
    )

    private val kernelCache = HashMap<String, List<Triple<Int, Int, Float>>>()

    private fun diskKernel(radius: Int): List<Triple<Int, Int, Float>> =
        kernelCache.getOrPut("disk_$radius") {
            val pts = mutableListOf<Triple<Int, Int, Float>>()
            for (dy in -radius..radius) {
                for (dx in -radius..radius) {
                    val dist = sqrt((dx * dx + dy * dy).toDouble())
                    if (dist > radius) continue
                    val weight = (1.0 - (dist / radius)).toFloat()
                    pts.add(Triple(dx, dy, weight))
                }
            }
            pts
        }

    private fun ringOffsets(innerRadius: Int, outerRadius: Int): List<Pair<Int, Int>> =
        kernelCache.getOrPut("ring_${innerRadius}_$outerRadius") {
            val pts = mutableListOf<Triple<Int, Int, Float>>()
            for (dy in -outerRadius..outerRadius) {
                for (dx in -outerRadius..outerRadius) {
                    val dist = sqrt((dx * dx + dy * dy).toDouble())
                    if (dist < innerRadius || dist > outerRadius) continue
                    pts.add(Triple(dx, dy, 1f))
                }
            }
            pts
        }.map { it.first to it.second }

    // ── Public API ─────────────────────────────────────────────

    fun readFilledBubbles(standardizedBitmap: Bitmap, template: CardTemplate): Pair<List<Int>, Float> {
        val report = readBubbleGroups(standardizedBitmap, template)
        return Pair(report.allFilled, report.overallConfidence)
    }

    fun readBubbleGroupsDetailed(standardizedBitmap: Bitmap, template: CardTemplate): BubbleReport =
        readBubbleGroups(standardizedBitmap, template)

    fun renderAnnotatedImage(
        standardizedBitmap: Bitmap,
        template: CardTemplate,
        report: BubbleReport,
        qrText: String? = null
    ): Bitmap {
        val annotated = standardizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotated)

        val positions = template.bubblePositions
        val filledSet = report.allFilled.toSet()
        val overvoteSet = report.groups
            .filter { it.status == GroupStatus.OVERVOTE }
            .flatMap { it.filledIndices }
            .toSet()

        val strokePaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true }
        val warnPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
            color = Color.rgb(255, 165, 0)
        }

        positions.forEachIndexed { index, pos ->
            val color = if (index in filledSet) Color.rgb(0, 200, 0) else Color.rgb(220, 30, 30)
            strokePaint.color = color
            val radius = BUBBLE_RADIUS.toFloat() + 2f
            canvas.drawCircle(pos.x, pos.y, radius, strokePaint)
            if (index in overvoteSet) {
                canvas.drawCircle(pos.x, pos.y, radius + 6f, warnPaint)
            }
        }

        if (!qrText.isNullOrBlank()) {
            val bannerHeight = 34f
            val bannerPaint = Paint().apply { style = Paint.Style.FILL; color = Color.argb(170, 0, 0, 0) }
            canvas.drawRect(0f, 0f, annotated.width.toFloat(), bannerHeight, bannerPaint)
            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 20f
                isAntiAlias = true
                textAlign = Paint.Align.LEFT
            }
            canvas.drawText("QR: $qrText", 8f, bannerHeight - 9f, textPaint)
        }

        return annotated
    }

    // ─── Core reading ──────────────────────────────────────────

    private fun readBubbleGroups(cleaned: Bitmap, template: CardTemplate): BubbleReport {
        val flattened = flattenIllumination(cleaned)
        val w = flattened.width
        val h = flattened.height
        val pixels = IntArray(w * h)
        flattened.getPixels(pixels, 0, w, 0, 0, w, h)
        if (flattened !== cleaned) flattened.recycle()

        val positions = template.bubblePositions
        val groups = template.bubbleGroups ?: listOf(positions.indices.first..positions.indices.last)

        val reads = positions.mapIndexed { index, nominal -> readSingleBubble(pixels, w, h, nominal, index) }
        val groupResults = groups.map { range -> evaluateGroup(range, reads) }

        val allFilled = groupResults.flatMap { it.filledIndices }
        val overallConfidence = if (groupResults.isNotEmpty()) {
            groupResults.map { it.confidence }.average().toFloat()
        } else 0f

        return BubbleReport(groupResults, allFilled, overallConfidence)
    }

    private fun readSingleBubble(pixels: IntArray, w: Int, h: Int, nominal: PointF, index: Int): BubbleRead {
        val background = sampleRingMedian(pixels, w, h, nominal.x.toInt(), nominal.y.toInt())

        var bestCenter = Pair(nominal.x.toInt(), nominal.y.toInt())
        var bestIntensity = sampleDisk(pixels, w, h, bestCenter.first, bestCenter.second)

        var dy = -REFINE_SEARCH_RADIUS
        while (dy <= REFINE_SEARCH_RADIUS) {
            var dx = -REFINE_SEARCH_RADIUS
            while (dx <= REFINE_SEARCH_RADIUS) {
                if (dx != 0 || dy != 0) {
                    val cx = nominal.x.toInt() + dx
                    val cy = nominal.y.toInt() + dy
                    val intensity = sampleDisk(pixels, w, h, cx, cy)
                    if (intensity < bestIntensity) {
                        bestIntensity = intensity
                        bestCenter = Pair(cx, cy)
                    }
                }
                dx += REFINE_SEARCH_STEP
            }
            dy += REFINE_SEARCH_STEP
        }

        if (background - bestIntensity > CENTROID_REFINE_MIN_SIGNAL) {
            val (rx, ry) = refineCentroid(pixels, w, h, bestCenter.first, bestCenter.second, background)
            if (rx != bestCenter.first || ry != bestCenter.second) {
                val refinedIntensity = sampleDisk(pixels, w, h, rx, ry)
                if (refinedIntensity <= bestIntensity) {
                    bestCenter = Pair(rx, ry)
                    bestIntensity = refinedIntensity
                }
            }
        }

        val normalizedDarkness = ((background - bestIntensity) / background.coerceAtLeast(1f)).coerceIn(0f, 1f)
        val fillRatio = sampleFillRatio(pixels, w, h, bestCenter.first, bestCenter.second, background)

        val combinedScore = normalizedDarkness * DARKNESS_WEIGHT + fillRatio * FILL_WEIGHT

        return BubbleRead(
            index = index,
            normalizedDarkness = combinedScore,
            rawIntensity = bestIntensity,
            localBackground = background,
            fillRatio = fillRatio
        )
    }

    private fun refineCentroid(pixels: IntArray, width: Int, height: Int, cx: Int, cy: Int, background: Float): Pair<Int, Int> {
        val searchRadius = BUBBLE_RADIUS + REFINE_SEARCH_RADIUS
        var sumX = 0.0
        var sumY = 0.0
        var sumW = 0.0
        for (dy in -searchRadius..searchRadius) {
            for (dx in -searchRadius..searchRadius) {
                if (dx * dx + dy * dy > searchRadius * searchRadius) continue
                val x = cx + dx
                val y = cy + dy
                if (x < 0 || x >= width || y < 0 || y >= height) continue
                val gray = ((pixels[y * width + x] shr 16) and 0xFF).toFloat()
                val darkness = (background - gray).coerceAtLeast(0f)
                sumX += darkness * x
                sumY += darkness * y
                sumW += darkness
            }
        }
        if (sumW < 1.0) return Pair(cx, cy)
        val refinedX = (sumX / sumW).toInt()
        val refinedY = (sumY / sumW).toInt()
        val maxShift = REFINE_SEARCH_RADIUS
        val clampedX = cx + (refinedX - cx).coerceIn(-maxShift, maxShift)
        val clampedY = cy + (refinedY - cy).coerceIn(-maxShift, maxShift)
        return Pair(clampedX, clampedY)
    }

    private fun evaluateGroup(range: IntRange, reads: List<BubbleRead>): GroupResult {
        val groupReads = range.mapNotNull { idx -> reads.getOrNull(idx) }
        if (groupReads.isEmpty()) {
            return GroupResult(range, GroupStatus.EMPTY, emptyList(), 0f, emptyList())
        }

        val values = groupReads.map { it.normalizedDarkness }
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        val std = sqrt(variance).coerceAtLeast(1e-6)

        val filledIndices = mutableListOf<Int>()
        for (read in groupReads) {
            val z = (read.normalizedDarkness - mean) / std
            if (z > Z_THRESHOLD && read.normalizedDarkness > MIN_ABSOLUTE_DARKNESS) {
                filledIndices.add(read.index)
            }
        }

        val status = when (filledIndices.size) {
            0 -> GroupStatus.EMPTY
            1 -> GroupStatus.SINGLE
            else -> GroupStatus.OVERVOTE
        }

        val confidence = if (filledIndices.size == 1) {
            val top = groupReads.maxByOrNull { it.normalizedDarkness }!!
            val second = groupReads.filter { it.index != top.index }.maxByOrNull { it.normalizedDarkness }
            val margin = if (second != null) top.normalizedDarkness - second.normalizedDarkness else top.normalizedDarkness
            margin.coerceIn(0f, 1f)
        } else 0f

        return GroupResult(range, status, filledIndices, confidence, groupReads)
    }

    private fun sampleDisk(pixels: IntArray, width: Int, height: Int, cx: Int, cy: Int): Float =
        sampleWeighted(pixels, width, height, cx, cy, diskKernel(BUBBLE_RADIUS))

    private fun sampleFillRatio(pixels: IntArray, width: Int, height: Int, cx: Int, cy: Int, background: Float): Float {
        val kernel = diskKernel(INNER_FILL_RADIUS)
        if (kernel.isEmpty()) return 0f
        val inkThreshold = background - (background * INK_RELATIVE_DROP).coerceAtLeast(INK_MIN_ABSOLUTE_DROP)
        var darkWeight = 0.0
        var totalWeight = 0.0
        for ((dx, dy, weight) in kernel) {
            val x = cx + dx
            val y = cy + dy
            if (x < 0 || x >= width || y < 0 || y >= height) continue
            val gray = ((pixels[y * width + x] shr 16) and 0xFF).toFloat()
            totalWeight += weight
            if (gray <= inkThreshold) darkWeight += weight
        }
        return if (totalWeight > 0) (darkWeight / totalWeight).toFloat() else 0f
    }

    private fun sampleRingMedian(pixels: IntArray, width: Int, height: Int, cx: Int, cy: Int): Float {
        val ring = ringOffsets(RING_INNER, RING_OUTER)
        if (ring.isEmpty()) return 255f
        val samples = ArrayList<Float>(ring.size)
        for ((dx, dy) in ring) {
            val x = cx + dx
            val y = cy + dy
            if (x < 0 || x >= width || y < 0 || y >= height) continue
            val pixel = pixels[y * width + x]
            samples.add(((pixel shr 16) and 0xFF).toFloat())
        }
        if (samples.isEmpty()) return 255f
        samples.sort()
        return samples[samples.size / 2]
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
            val gray = ((pixel shr 16) and 0xFF).toDouble()
            sum += gray * weight
            weightSum += weight
        }
        return if (weightSum > 0) (sum / weightSum).toFloat() else 255f
    }

    // ─── Illumination flattening ─────────────────────────────────

    private fun flattenIllumination(bitmap: Bitmap): Bitmap {
        if (!ENABLE_ILLUMINATION_FLATTENING) return bitmap

        val src = Mat()
        val gray = Mat()
        val gray32 = Mat()
        val illumination = Mat()
        val corrected32 = Mat()
        val corrected8 = Mat()
        val dest = Mat()
        return try {
            Utils.bitmapToMat(bitmap, src)
            if (src.channels() > 1) Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY) else src.copyTo(gray)
            gray.convertTo(gray32, CvType.CV_32F)

            var kernelSize = bitmap.width / 6
            if (kernelSize < 31) kernelSize = 31
            if (kernelSize % 2 == 0) kernelSize += 1

            Imgproc.GaussianBlur(gray32, illumination, Size(kernelSize.toDouble(), kernelSize.toDouble()), 0.0)

            Core.max(illumination, Scalar(10.0), illumination)

            val meanIllum = Core.mean(illumination).`val`[0]
            if (meanIllum <= 1.0) return bitmap

            Core.divide(gray32, illumination, corrected32, meanIllum)
            Core.min(corrected32, Scalar(255.0), corrected32)
            corrected32.convertTo(corrected8, CvType.CV_8U)

            Imgproc.cvtColor(corrected8, dest, Imgproc.COLOR_GRAY2RGBA)
            val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(dest, result)
            result
        } catch (e: Exception) {
            Log.w(TAG, "Illumination flattening failed, using original image", e)
            bitmap
        } finally {
            src.release(); gray.release(); gray32.release()
            illumination.release(); corrected32.release(); corrected8.release(); dest.release()
        }
    }
}