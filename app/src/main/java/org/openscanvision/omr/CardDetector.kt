// CardDetector.kt
package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object CardDetector {
    private const val TAG = "CardDetector"
    private const val MIN_AREA = 15000
    private const val MAX_AREA_RATIO = 0.85
    private const val EPSILON_FACTOR = 0.02

    private const val MARKER_SEARCH_MARGIN_PX = 40
    private const val MARKER_DARKNESS_THRESHOLD = 80
    private const val MARKER_MIN_DARK_PIXELS = 15

    fun detectCardCorners(bitmap: Bitmap): List<PointF>? {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        if (src.empty()) return null

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)

        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(blurred, edges, 50.0, 150.0)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        val dilated = Mat()
        Imgproc.dilate(edges, dilated, kernel)
        kernel.release()

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(dilated, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imgArea = src.width() * src.height()
        var bestContour2f: MatOfPoint2f? = null
        var maxArea = 0.0

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < MIN_AREA || area > imgArea * MAX_AREA_RATIO) continue

            val contour2f = MatOfPoint2f()
            contour2f.fromArray(*contour.toArray())
            val peri = Imgproc.arcLength(contour2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, peri * EPSILON_FACTOR, true)
            contour2f.release()

            if (approx.rows() == 4 && area > maxArea) {
                maxArea = area
                bestContour2f = MatOfPoint2f()
                approx.copyTo(bestContour2f)
            }
            approx.release()
        }

        src.release(); gray.release(); blurred.release(); edges.release(); dilated.release(); hierarchy.release()
        contours.forEach { it.release() }

        bestContour2f ?: return null
        val points = bestContour2f!!.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        bestContour2f!!.release()
        if (points.size != 4) return null

        val tl = points.minByOrNull { it.x + it.y }!!
        val br = points.maxByOrNull { it.x + it.y }!!
        val remaining = points.filter { it != tl && it != br }
        val tr = remaining.minByOrNull { it.x - it.y }!!
        val bl = remaining.maxByOrNull { it.x - it.y }!!

        return listOf(tl, tr, br, bl)
    }

    fun predictImagePoints(templatePoints: List<PointF>, templateToImage: Matrix): List<PointF> {
        val coords = FloatArray(templatePoints.size * 2)
        templatePoints.forEachIndexed { i, p ->
            coords[i * 2] = p.x
            coords[i * 2 + 1] = p.y
        }
        val mapped = coords.copyOf()
        templateToImage.mapPoints(mapped)
        return templatePoints.indices.map { PointF(mapped[it * 2], mapped[it * 2 + 1]) }
    }

    fun detectRefMarkersNearPredicted(
        bitmap: Bitmap,
        predictedPositions: List<PointF>
    ): List<PointF?> {
        val results = mutableListOf<PointF?>()

        for (pred in predictedPositions) {
            val left = (pred.x - MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.width - 1)
            val top = (pred.y - MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.height - 1)
            val right = (pred.x + MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.width - 1)
            val bottom = (pred.y + MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.height - 1)
            val w = right - left
            val h = bottom - top

            if (w <= 0 || h <= 0) {
                results.add(null)
                continue
            }

            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, left, top, w, h)

            var sumX = 0.0
            var sumY = 0.0
            var count = 0
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val gray = android.graphics.Color.red(pixels[y * w + x])
                    if (gray < MARKER_DARKNESS_THRESHOLD) {
                        sumX += x
                        sumY += y
                        count++
                    }
                }
            }

            if (count < MARKER_MIN_DARK_PIXELS) {
                results.add(null)
            } else {
                results.add(PointF((left + sumX / count).toFloat(), (top + sumY / count).toFloat()))
            }
        }
        return results
    }

    fun buildRefinedHomography(
        templatePoints: List<PointF>,
        imagePoints: List<PointF>
    ): Matrix? {
        if (templatePoints.size != imagePoints.size || templatePoints.size < 4) {
            Log.w(TAG, "Need >=4 matched correspondences, got ${templatePoints.size}")
            return null
        }

        val src = MatOfPoint2f(*templatePoints.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val dst = MatOfPoint2f(*imagePoints.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())

        val hMat = try {
            Calib3d.findHomography(src, dst, 0)
        } catch (e: Exception) {
            Log.e(TAG, "findHomography failed", e)
            src.release(); dst.release()
            return null
        }
        src.release(); dst.release()

        if (hMat.empty()) return null

        val values = FloatArray(9)
        for (r in 0..2) {
            for (c in 0..2) {
                values[r * 3 + c] = hMat.get(r, c)[0].toFloat()
            }
        }
        hMat.release()

        val matrix = Matrix()
        matrix.setValues(values)
        return matrix
    }
}